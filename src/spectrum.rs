pub mod eq;
pub mod art;
pub mod aslt;
pub mod gpu;
pub mod gpu_calib;
pub mod freq_scale;
pub mod channels;
pub mod timing;

use egui::{Color32, Pos2, Rect, Shape, Stroke};
use serde::{Deserialize, Serialize};
use rustfft::{FftPlanner, num_complex::Complex};
use rodio::Source;
use std::f32::consts::PI;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eq::{BandKind, EqBand, EqOverlayMode, EqState, EqStateHandle,
         EqPreset, PresetScope, PresetRef, EqPresetLibrary, eq_band_color,
         eq_biquad_coeffs, biquad_response_db,
         total_eq_response_db, bands_equal, KWeightFilter};
use art::{ArtFit, ArtSpectrumMode, ArtMaskMode, ArtDisplaySettings,
          ArtSettingsStore, art_dest_rect, draw_bars_art_mask};

pub const DEFAULT_FFT_SIZE: usize = 8192;
pub const DEFAULT_BAR_COUNT: usize = 1024;
pub const MIN_BAR_COUNT: usize = 16;
pub const MAX_BAR_COUNT: usize = 1024;
/// Pre-process frame rate — how many rows a second the cached analysis holds.
///
/// The renderer does **not** interpolate between rows; it holds the most
/// recently crossed one until the next arrives (zero-order hold). This comment
/// used to claim interpolation, which made the setting look like a quality
/// ceiling when it is really the temporal resolution itself. Cost scales
/// linearly with it, and every row is now consumed exactly once — see
/// [`timing`] — so raising it buys motion detail rather than discarded work.
pub const DEFAULT_PRE_FPS: f32 = 180.0;
pub const DEFAULT_MIN_FREQ: f32 = 20.0;
pub const DEFAULT_MAX_FREQ: f32 = 24_000.0;
/// Fallback ring depth, used only before a row rate is known.
///
/// The depth is chosen at runtime from the wanted span and the rate rows are
/// expected to arrive at — see [`timing::waterfall_rows_for`] — because rows per
/// second differ between modes and settings, and a fixed row count therefore
/// means a different amount of time on every configuration.
const WATERFALL_ROWS_FALLBACK: usize = 120;
// ---------------------------------------------------------------------------
// New public enums: loudness mode, window function
// ---------------------------------------------------------------------------
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum LoudnessMode {
    /// Raw dB magnitude — no psychoacoustic correction.
    Flat,
    /// ISO 226:2003 equal-loudness weighting at 40 phon.
    EqualLoudness,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum WindowFn { Hann, Hamming, Blackman, FlatTop }

/// Sub-bin interpolation method used when mapping FFT bins to spectrum bars.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum InterpolationMode {
    /// No interpolation — nearest bin value. Fastest, most "honest".
    None,
    /// Linear blend between adjacent bins. Fast but angular in the low end.
    Linear,
    /// Smooth cubic spline through neighbours. Good balance of quality and speed.
    CatmullRom,
    /// Monotone cubic (Fritsch-Carlson). No overshoot, shape-preserving.
    Pchip,
    /// Akima local cubic. Smooth without oscillation. Designed for scientific data.
    Akima,
    /// Sinc-windowed (a=3). Best frequency accuracy, but can ring near sharp peaks.
    Lanczos,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum BarMappingMode {
    /// Flat overlap average — equal weight to all bins in range.
    FlatOverlap,
    /// Gaussian-weighted average — bins near bar centre frequency weighted more heavily.
    Gaussian,
    /// Constant-Q transform — Hann kernel whose bandwidth scales with frequency (f/Q).
    /// Each bar has the same relative frequency resolution regardless of pitch.
    Cqt,
    /// Adaptive Superlet Transform — not an FFT mapping at all, but a direct
    /// wavelet analysis of the time-domain signal (see [`aslt`]).
    ///
    /// Pre-process only. The live path has ~2 ms per frame to work with and a
    /// superlet needs orders of magnitude more, so real-time display keeps using
    /// the FFT and falls back to the CQT kernel below.
    Superlet,
}

/// dB reference that lines the superlet up with the FFT path.
///
/// The FFT path reports `20·log10(|X| / fft_size)`. A unit-amplitude sine
/// through a Hann window peaks at `fft_size/4` (coherent gain ½, and half the
/// energy goes to the negative frequency), so it lands at −12.04 dB. ASLT
/// magnitudes are normalised so the same sine reads 1.0, i.e. 0 dB. Without this
/// factor, switching bar-mapping mode would jump the whole display 12 dB.
const ASLT_DB_REF: f32 = 0.25;

// ---------------------------------------------------------------------------
// Peak hold
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum PeakDecayMode {
    /// Constant fall speed.
    Linear,
    /// Accelerates as it falls — feels physical.
    Gravity,
    /// Peak fades in place rather than falling.
    FadeOut,
}

impl Default for PeakDecayMode {
    fn default() -> Self { PeakDecayMode::Gravity }
}

#[derive(Clone)]
pub struct PeakHoldConfig {
    pub enabled:         bool,
    /// How long the peak marker freezes before decaying (ms).
    pub hold_ms:         f32,
    /// Initial decay speed (normalized units/sec).
    pub fall_speed:      f32,
    /// Fall acceleration for Gravity mode (normalized units/sec²).
    pub acceleration:    f32,
    pub decay_mode:      PeakDecayMode,
    /// Height of the peak marker in physical pixels.
    pub peak_thickness:  u8,
    pub color:           Color32,
}

impl Default for PeakHoldConfig {
    fn default() -> Self {
        Self {
            enabled:        true,
            hold_ms:        500.0,
            fall_speed:     3.0,
            acceleration:   4.0,
            decay_mode:     PeakDecayMode::Gravity,
            peak_thickness: 2,
            color:          Color32::WHITE,
        }
    }
}

// Progress is split into three named stages rather than one 0–100 ramp, because
// the middle one used to look like a hang: decoding reported 0–49, the transform
// reported 50–99, and the loudness/chroma pass between them reported nothing at
// all, so the bar sat frozen at exactly 50 % for as long as that pass took.
const PROG_DECODE_END: usize = 39;
const PROG_LOUDNESS_END: usize = 49;

/// Which stage a raw progress percentage belongs to. Named so a stalled-looking
/// number is at least an *explained* stalled-looking number.
fn phase_label(pct: usize) -> &'static str {
    match pct {
        0..=PROG_DECODE_END   => "Decoding…",
        40..=PROG_LOUDNESS_END => "Loudness/key…",
        100                    => "Finishing…",
        _                      => "Transform…",
    }
}

/// Store `v` only if it beats what is already there.
///
/// The transform reports from many rayon threads at once. Each reads a running
/// total and then stores a percentage, and nothing stops a thread that read an
/// older total from storing after one that read a newer total — which shows up
/// as a progress bar that twitches backwards.
fn store_progress_max(progress: &Arc<AtomicUsize>, v: usize) {
    let mut cur = progress.load(Ordering::Relaxed);
    while v > cur {
        match progress.compare_exchange_weak(cur, v, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}

/// Coarse "time remaining" wording. Rounded hard on purpose — a superlet ETA
/// is an extrapolation, and second-level precision would imply it is not.
fn fmt_eta(secs: usize) -> String {
    match secs {
        0..=5           => "a few seconds".to_string(),
        6..=89          => format!("{secs}s"),
        90..=3599       => format!("{}m", (secs + 30) / 60),
        _               => {
            let h = secs / 3600;
            let m = (secs % 3600 + 30) / 60;
            if m == 0 { format!("{h}h") } else { format!("{h}h {m}m") }
        }
    }
}

/// Log-spaced centre frequency for bar index `i` of `n` bars in [min, max].
fn bar_center_freq(i: usize, n: usize, min_freq: f32, max_freq: f32) -> f32 {
    let t = (i as f32 + 0.5) / n as f32;
    10f32.powf(min_freq.log10() + t * (max_freq.log10() - min_freq.log10()))
}

/// Build a window coefficient vector of `size` samples for the given function.
pub fn make_window(size: usize, wfn: &WindowFn) -> Vec<f32> {
    let n = size as f32 - 1.0;
    (0..size).map(|i| {
        let t = i as f32;
        match wfn {
            WindowFn::Hann     => 0.5 * (1.0 - (2.0 * PI * t / n).cos()),
            WindowFn::Hamming  => 0.54 - 0.46 * (2.0 * PI * t / n).cos(),
            WindowFn::Blackman => 0.42 - 0.5 * (2.0 * PI * t / n).cos()
                                       + 0.08 * (4.0 * PI * t / n).cos(),
            WindowFn::FlatTop  => 1.0
                - 1.930 * (2.0 * PI * t / n).cos()
                + 1.290 * (4.0 * PI * t / n).cos()
                - 0.388 * (6.0 * PI * t / n).cos()
                + 0.028 * (8.0 * PI * t / n).cos(),
        }
    }).collect()
}

// ---------------------------------------------------------------------------
// ISO 226:2003 — compute SPL at `freq` Hz needed for 40 phons.
//
// Parameters from ISO 226:2003, Table 1.  Valid range: 20 Hz – 12 500 Hz.
// Outside that range the nearest endpoint is used (extrapolation clamped).
// ---------------------------------------------------------------------------

fn iso226_spl_40phon(freq: f32) -> f32 {
    // (frequency Hz, alpha_f, L_u dB, T_f dB)
    const T: &[(f32, f32, f32, f32)] = &[
        (20.0,    0.532, -31.6, 78.5),
        (25.0,    0.506, -27.2, 68.7),
        (31.5,    0.480, -23.0, 59.5),
        (40.0,    0.455, -19.1, 51.1),
        (50.0,    0.432, -15.9, 44.0),
        (63.0,    0.409, -13.0, 37.5),
        (80.0,    0.387, -10.3, 31.5),
        (100.0,   0.367,  -8.1, 26.5),
        (125.0,   0.349,  -6.2, 22.1),
        (160.0,   0.330,  -4.5, 17.9),
        (200.0,   0.315,  -3.1, 14.4),
        (250.0,   0.301,  -2.0, 11.4),
        (315.0,   0.288,  -1.1,  8.6),
        (400.0,   0.276,  -0.4,  6.2),
        (500.0,   0.267,   0.0,  4.4),
        (630.0,   0.259,   0.3,  3.0),
        (800.0,   0.253,   0.5,  2.2),
        (1000.0,  0.250,   0.0,  2.4),
        (1250.0,  0.246,  -2.7,  3.5),
        (1600.0,  0.244,  -4.1,  1.7),
        (2000.0,  0.243,   1.0, -1.3),
        (2500.0,  0.243,   6.6, -4.2),
        (3150.0,  0.243,  15.3, -6.0),
        (4000.0,  0.242,  21.7, -5.4),
        (5000.0,  0.242,  25.0, -1.5),
        (6300.0,  0.245,  27.5,  4.3),
        (8000.0,  0.254,  30.0, 12.7),
        (10000.0, 0.271,  36.5, 21.2),
        (12500.0, 0.301,  40.0, 35.8),
    ];

    let freq = freq.clamp(T[0].0, T[T.len() - 1].0);
    let lf = freq.log10();

    // Binary search for the surrounding pair; interpolate on log-frequency axis.
    let i = T.partition_point(|&(f, ..)| f.log10() < lf);
    let (alpha_f, l_u, t_f) = if i == 0 {
        let (_, a, l, t) = T[0]; (a, l, t)
    } else if i >= T.len() {
        let (_, a, l, t) = T[T.len() - 1]; (a, l, t)
    } else {
        let (f0, a0, lu0, tf0) = T[i - 1];
        let (f1, a1, lu1, tf1) = T[i];
        let s = (lf - f0.log10()) / (f1.log10() - f0.log10());
        (a0 + s * (a1 - a0), lu0 + s * (lu1 - lu0), tf0 + s * (tf1 - tf0))
    };

    // ISO 226:2003 Eq. (2) for L_N = 40 phon
    let ln = 40.0_f32;
    let a_f = 4.47e-3 * (10_f32.powf(0.025 * ln) - 1.15)
        + (0.4 * 10_f32.powf((t_f + l_u) / 10.0 - 9.0)).powf(alpha_f);
    (10.0 / alpha_f) * a_f.log10() - l_u + 94.0
}

/// Build a per-bar equal-loudness correction vector (dB offset added to raw dB).
/// Positive at 3–5 kHz (ear very sensitive there), negative in bass and highs.
pub fn compute_eq_weights(
    n_bars: usize, min_freq: f32, max_freq: f32, scale: freq_scale::FreqScale,
) -> Vec<f32> {
    let ref_spl = iso226_spl_40phon(1000.0); // ≈ 40.0 by definition
    let mut w: Vec<f32> = (0..n_bars)
        .map(|b| {
            // Read at the frequency bar `b` actually sits at, which is what the
            // scale decides. A weight computed on the log axis and applied to
            // ERB-spaced bars is the wrong correction for every bar.
            let freq = scale.bar_center(b, n_bars, min_freq, max_freq);
            let freq_clamped = freq.clamp(20.0, 12_500.0);
            ref_spl - iso226_spl_40phon(freq_clamped)
        })
        .collect();

    // Referenced to 1 kHz the curve *boosts* by up to +22 dB across 2-10 kHz,
    // where the ear is most sensitive. Bar heights arrive normalised to 0..1
    // over an 80 dB range, so a positive correction has nowhere to go: every bar
    // in that region pinned to the ceiling and stopped moving, a static block an
    // octave and a half wide.
    //
    // Sliding the whole curve down so its peak sits at 0 dB fixes the clipping
    // and is wrong in the other direction: it drags 1 kHz down 22 dB and the
    // bass past the floor, so the entire display goes dim. That was tried and
    // is what "everything looks really quiet" was.
    //
    // Clamping the positive lobe instead keeps 1 kHz where it has always been
    // and costs only the boost: 2-10 kHz reads exactly as it does with weighting
    // off, and everything below 1 kHz carries the ISO tilt in full. The tilt is
    // the informative half — it is what makes bass look as quiet as it sounds —
    // and on a relative display, giving up the boost costs far less than pulling
    // every bar down by a fixed 22 dB.
    for v in &mut w { *v = v.min(0.0); }
    w
}

#[cfg(test)]
mod nyquist_tests {
    use super::*;

    /// A 44.1 kHz track with the default 24 kHz ceiling asks for bars above
    /// Nyquist. Every bar must still resolve to a bin that exists.
    ///
    /// This killed the process on the first 44.1 kHz file played with a
    /// non-constant-Q mapping: 48 kHz material has Nyquist exactly at the
    /// default `max_freq`, so nothing above the ceiling was ever requested and
    /// the missing bound stayed invisible.
    #[test]
    fn bars_above_nyquist_stay_inside_the_spectrum() {
        let mut a = SpectrumAnalyzer::new(new_sample_buf());
        a.sample_rate = 44_100;
        a.min_freq = DEFAULT_MIN_FREQ;
        a.max_freq = DEFAULT_MAX_FREQ; // 24 kHz — above Nyquist here
        let padded = a.fft_size * 2;
        let buf = vec![Complex { re: 0.5f32, im: 0.0 }; padded];

        // The constant-Q branch always clamped; the others are the ones that
        // reached the unguarded index, and every interpolation mode routes
        // through it.
        for mapping in [BarMappingMode::Superlet, BarMappingMode::FlatOverlap,
                        BarMappingMode::Gaussian, BarMappingMode::Cqt] {
            for interp in [InterpolationMode::None, InterpolationMode::Linear,
                           InterpolationMode::CatmullRom, InterpolationMode::Pchip] {
                a.bar_mapping = mapping.clone();
                a.interp_mode = interp.clone();
                let bars = a.bins_to_bars(&buf, a.sample_rate, 1024, padded);
                assert_eq!(bars.len(), 1024, "{mapping:?}/{interp:?}");
                assert!(bars.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
                        "{mapping:?}/{interp:?} produced a value outside 0..1");
            }
        }
    }

    /// The unguarded index itself, at and past the end.
    #[test]
    fn sub_bin_interpolation_holds_at_the_edges() {
        let norms = [1.0f32, 2.0, 3.0, 4.0];
        for mode in [InterpolationMode::None, InterpolationMode::Linear,
                     InterpolationMode::CatmullRom, InterpolationMode::Pchip] {
            for &c in &[-5.0f32, -0.4, 0.0, 1.5, 3.0, 3.9, 4.0, 99.0] {
                let v = interp_sub_bin(&norms, c, &mode);
                assert!(v.is_finite(), "{mode:?} at {c} gave {v}");
            }
        }
        assert_eq!(interp_sub_bin(&norms, 2.0, &InterpolationMode::None), 3.0);
    }
}

#[cfg(test)]
mod iso226_tests {
    use super::*;

    /// The curve itself, against the published 40-phon values.
    #[test]
    fn matches_the_standard_at_known_points() {
        // ISO 226:2003 Table 1: the 40 phon contour passes through 40 dB SPL at
        // 1 kHz by definition, and ~99.85 dB at 20 Hz.
        assert!((iso226_spl_40phon(1000.0) - 40.0).abs() < 0.1);
        assert!((iso226_spl_40phon(20.0) - 99.85).abs() < 0.5);
        // Most sensitive around 3–4 kHz, where it dips well below 40.
        assert!(iso226_spl_40phon(3150.0) < 30.0);
        assert!(iso226_spl_40phon(3150.0) < iso226_spl_40phon(1000.0));
    }

    /// No weight may be positive.
    ///
    /// Bar heights reach the display as 0..1 over an 80 dB range, so a positive
    /// correction has nowhere to go: referenced to 1 kHz this curve boosts by up
    /// to +22 dB across 2–10 kHz, which pinned every bar in that octave-and-a-half
    /// to the ceiling and held it there. A static block from 2 kHz to 10 kHz is
    /// what that looked like on screen.
    #[test]
    fn weights_never_boost_so_nothing_can_saturate() {
        let w = compute_eq_weights(1024, 20.0, 24_000.0, freq_scale::FreqScale::Log);
        assert_eq!(w.len(), 1024);
        assert!(w.iter().all(|&v| v <= 1e-3), "no bar may be boosted");

        let at = |hz: f32| {
            let t = (hz.log10() - 20f32.log10()) / (24_000f32.log10() - 20f32.log10());
            w[((t * 1024.0) as usize).min(1023)]
        };

        // 1 kHz keeps full height. Referencing the curve to its own peak instead
        // satisfies "never boosts" just as well and drags every bar down 22 dB
        // with it — the display goes uniformly dim and the tilt is no easier to
        // read. Whatever else changes here, this must not.
        assert!(at(1000.0) > -1.0, "1 kHz must stay at full height, got {}", at(1000.0));
        assert!(at(3500.0) > -1.0, "the boosted region flattens rather than lifting");

        // The tilt below 1 kHz is the informative half and survives in full.
        assert!(at(1000.0) > at(200.0), "1 kHz should sit above 200 Hz");
        assert!(at(200.0) > at(60.0), "200 Hz should sit above 60 Hz");
        assert!(at(60.0) < -20.0, "bass should be clearly attenuated, got {}", at(60.0));
    }
}

// ---------------------------------------------------------------------------
// Chromagram, key detection, BPM
// ---------------------------------------------------------------------------

const NOTE_NAMES: &[&str] = &["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];

/// Krumhansl-Schmuckler key profiles (major / minor).
const KS_MAJOR: [f32; 12] = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88];
const KS_MINOR: [f32; 12] = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17];

/// Convert FFT half-spectrum → 12-bin chromagram (pitch-class energy sums).
/// Feeds key detection (`detect_key`).
///
/// Key improvements over the naive approach:
/// - Frequency range 110 Hz–4186 Hz (A2–C8): covers note fundamentals and
///   their first few harmonics without being swamped by sub-bass or high noise.
/// - Log-magnitude compression: `log2(1 + v*scale)` equalises the energy
///   contribution across octaves; without this, bass bins dominate.
/// - Returns unnormalised values so callers can accumulate across multiple
///   frames (energy-weighted) before normalising.
fn compute_chroma(norms: &[f32], sr: u32, fft_size: usize) -> [f32; 12] {
    let mut chroma = [0.0f32; 12];
    let scale = fft_size as f32;
    for (i, &v) in norms.iter().enumerate().skip(1) {
        let freq = i as f32 * sr as f32 / fft_size as f32;
        if !(110.0..=4_186.0).contains(&freq) { continue; }
        // Log-magnitude compression to de-emphasise dominant low-freq bins
        let log_v = (1.0 + v * scale).log2();
        let pc = ((12.0 * (freq / 261.63_f32).log2()).round() as i32).rem_euclid(12) as usize;
        chroma[pc] += log_v;
    }
    chroma
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 { 0.0 } else { dot / (na * nb) }
}

/// Identify the best-matching major/minor key from an accumulated chromagram.
/// Uses fixed-size stack arrays — zero heap allocation.
fn detect_key(chroma: &[f32; 12]) -> String {
    let mut best_score = -2.0f32;
    let mut best_name = String::new();
    for root in 0..12usize {
        let maj: [f32; 12] = std::array::from_fn(|i| KS_MAJOR[(i + 12 - root) % 12]);
        let min: [f32; 12] = std::array::from_fn(|i| KS_MINOR[(i + 12 - root) % 12]);
        let ms = cosine_sim(chroma, &maj);
        let mn = cosine_sim(chroma, &min);
        if ms > best_score { best_score = ms; best_name = format!("{} major", NOTE_NAMES[root]); }
        if mn > best_score { best_score = mn; best_name = format!("{} minor", NOTE_NAMES[root]); }
    }
    best_name
}

/// Estimate BPM from a spectral-flux series (at `flux_rate` Hz) via autocorrelation.
fn detect_bpm(flux: &[f32], flux_rate: f32) -> f32 {
    if flux.len() < 32 { return 0.0; }
    let lag_min = ((flux_rate * 60.0 / 180.0) as usize).max(1);
    let lag_max = ((flux_rate * 60.0 / 60.0) as usize).min(flux.len() / 2);
    if lag_min >= lag_max { return 0.0; }
    let mean = flux.iter().sum::<f32>() / flux.len() as f32;
    let fc: Vec<f32> = flux.iter().map(|&x| x - mean).collect();
    let (mut best_lag, mut best_acf) = (lag_min, f32::NEG_INFINITY);
    for lag in lag_min..=lag_max {
        let n = fc.len() - lag;
        let acf = fc[..n].iter().zip(&fc[lag..]).map(|(a, b)| a * b).sum::<f32>() / n as f32;
        if acf > best_acf { best_acf = acf; best_lag = lag; }
    }
    if best_acf <= 0.0 { return 0.0; }
    (60.0 * flux_rate / best_lag as f32).round()
}

// ---------------------------------------------------------------------------
// Sample conversion (works with both i16 and f32 rodio decoders)
// ---------------------------------------------------------------------------

pub trait SampleToF32: Copy + Send + 'static {
    fn to_spectrum_f32(self) -> f32;
}

impl SampleToF32 for i16 {
    fn to_spectrum_f32(self) -> f32 {
        self as f32 / 32_768.0
    }
}

impl SampleToF32 for f32 {
    fn to_spectrum_f32(self) -> f32 {
        self
    }
}

impl SampleToF32 for u16 {
    fn to_spectrum_f32(self) -> f32 {
        (self as f32 - 32_768.0) / 32_768.0
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum SpectrumMode {
    RealTime,
    PreProcess,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum VizStyle {
    Bars,
    Line,
    FilledArea,
    Waterfall,
    Phasescope,
    Spectrogram,
    OctaveBands,
}

/// Ring buffer shared between the audio thread (writer) and UI thread (reader).
pub type SampleBuf = Arc<Mutex<Vec<f32>>>;

pub fn new_sample_buf() -> SampleBuf {
    Arc::new(Mutex::new(Vec::with_capacity(DEFAULT_FFT_SIZE * 8)))
}

/// Longest analysis window any live path can ask for.
///
/// `auto_fft_size_for` climbs to 32768 at high sample rates and the FFT-size
/// selector offers it outright, so every buffer feeding a live analysis has to
/// hold at least this much. The stereo buffer used to be fixed at 8192 frames,
/// which was a quarter of one window: a 192 kHz track selected a 32768-point
/// FFT and the channel path would have had nothing like enough history to run
/// it. Both caps are now derived from here so they cannot drift apart again.
pub const MAX_ANALYSIS_WINDOW: usize = 32_768;

/// How much history each live buffer keeps. Two windows, so an analysis can
/// always be served without waiting for a refill.
pub const MONO_CAP: usize = MAX_ANALYSIS_WINDOW * 2;
pub const STEREO_CAP: usize = MAX_ANALYSIS_WINDOW * 2;

// The property that matters, checked where it cannot be forgotten: a buffer
// that cannot hold one whole window can never serve an analysis at all.
const _: () = assert!(MONO_CAP >= MAX_ANALYSIS_WINDOW);
const _: () = assert!(STEREO_CAP >= MAX_ANALYSIS_WINDOW);

/// Stereo sample pairs, plus the facts needed to know what they are.
///
/// The frames alone cannot say whether they are the current track's: nothing
/// clears them when a stream ends, and native DSD attaches no tap at all, so a
/// non-empty buffer is not evidence of a live stereo source. `channels` is set
/// from the decoder or device format by whoever builds the tap, and is
/// [`channels::NO_LIVE_TAP`] whenever nothing is writing.
///
/// `generation` is bumped every time a stream starts or ends. A tap captures it
/// at construction and refuses to write once it no longer matches, which closes
/// the window where a previous stream's last flush could land after the new
/// track had already reset the buffer.
pub struct StereoTapBuf {
    pub frames: Vec<[f32; 2]>,
    pub channels: u16,
    pub generation: u64,
    /// Which route/session the buffer currently belongs to.
    ///
    /// Distinct from `generation`, and the distinction is the whole point.
    /// `generation` says *who is writing*; `route_epoch` says *which playback
    /// session is entitled to write at all*. Only the engine advances it, once
    /// per route teardown.
    ///
    /// A source is constructed when it is appended and claims when it is first
    /// pulled, and those can be seconds apart. `Sink::stop` only sets an atomic
    /// that the mixer notices asynchronously, so a render callback that has
    /// already passed the stop check can still enter a source belonging to a
    /// session the engine has finished with. Without an epoch that late first
    /// pull would clear the buffer and take ownership from whatever route had
    /// started in the meantime — a claim is unconditional by construction, so
    /// generation alone cannot refuse it.
    pub route_epoch: u64,
}

pub type StereoBuf = Arc<Mutex<StereoTapBuf>>;

pub fn new_stereo_buf() -> StereoBuf {
    Arc::new(Mutex::new(StereoTapBuf {
        frames: Vec::with_capacity(STEREO_CAP),
        channels: channels::NO_LIVE_TAP,
        generation: 0,
        route_epoch: 0,
    }))
}

/// Prepare both analyser buffers for a tap, and report the route it belongs to.
///
/// One helper because there is one live-PCM route, not a mono one and a stereo
/// one. The previous shape reserved and claimed the stereo buffer under a lease
/// and left the mono buffer entirely unguarded — so a source whose session had
/// ended still fed the Mix, which is the series most people are looking at.
///
/// Blocking and allocating, deliberately: this runs on the thread assembling
/// the chain. Everything the tap does afterwards is non-blocking because the
/// growth happened here.
///
/// The lock order is stereo-then-mono, and it is the same order publication and
/// teardown use. There is only one order in this file and this is it.
pub fn prepare_live_tap(stereo_buf: &StereoBuf, sample_buf: &SampleBuf) -> u64 {
    let mut v = match stereo_buf.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let want = STEREO_CAP.saturating_sub(v.frames.len());
    v.frames.reserve(want);
    debug_assert!(v.frames.capacity() >= STEREO_CAP);
    let epoch = v.route_epoch;
    if let Ok(mut m) = sample_buf.lock() {
        let want = MONO_CAP.saturating_sub(m.len());
        m.reserve(want);
        debug_assert!(m.capacity() >= MONO_CAP);
    }
    epoch
}

/// A test-only observation point inside [`publish_live_pcm`].
///
/// Publication holds the ownership guard across both writes, so a teardown
/// cannot interleave with it. That is an invariant about lock scope, which no
/// amount of black-box poking can demonstrate — the only way to stand between
/// the two writes is to be called from between them.
#[cfg(test)]
pub mod publish_hook {
    use std::cell::RefCell;

    type Hook = Box<dyn Fn()>;

    thread_local! {
        static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
    }

    /// Run `f` between lease validation and the first buffer write, for as long
    /// as the returned guard lives.
    pub fn install(f: impl Fn() + 'static) -> Guard {
        HOOK.with(|h| *h.borrow_mut() = Some(Box::new(f)));
        Guard
    }

    pub struct Guard;

    impl Drop for Guard {
        fn drop(&mut self) {
            HOOK.with(|h| *h.borrow_mut() = None);
        }
    }

    pub(super) fn fire() {
        // Taken out and put back, so a hook that publishes again does not
        // recurse into itself.
        let hook = HOOK.with(|h| h.borrow_mut().take());
        if let Some(f) = hook {
            f();
            HOOK.with(|h| *h.borrow_mut() = Some(f));
        }
    }
}

/// Publish one batch of analysis PCM under `lease`, or publish nothing.
///
/// The only way a tap writes to either buffer. It exists because the two
/// buffers are one route and were being guarded as two: the bit-perfect tap
/// wrote mono *before* it checked its generation, and the shared source wrote
/// mono whatever its claim state was, so every teardown left the Mix showing a
/// track that had stopped.
///
/// The ownership state and the stereo frames live in the same mutex, so the
/// lease is validated and the stereo write performed under one guard; the mono
/// guard is taken while still holding it, in the one lock order this file uses.
///
/// Runs on an audio or render thread, so: `try_lock` only, no allocation (both
/// buffers were reserved by [`prepare_live_tap`], and `drain` and
/// `extend_from_slice` within capacity do not grow), no logging, and no
/// retrying. A busy lock drops this batch — the analyser is a display, and a
/// dropped batch costs a few milliseconds of history, where spinning costs the
/// audio.
///
/// Returns whether the batch was published. The caller clears its batches
/// either way; holding them back would show the display audio from the wrong
/// side of a discontinuity.
pub fn publish_live_pcm(
    stereo_buf: &StereoBuf,
    sample_buf: &SampleBuf,
    lease: u64,
    mono: &[f32],
    pairs: &[[f32; 2]],
) -> bool {
    let Ok(mut v) = stereo_buf.try_lock() else {
        return false;
    };
    if v.generation != lease {
        return false;
    }
    // A forced observation point, between validating the lease and touching
    // either buffer. It is where a teardown would have to interleave for a
    // publication to be half-applied, and the hook exists so a test can stand
    // there and demonstrate that it cannot.
    #[cfg(test)]
    publish_hook::fire();
    // Mono first, still holding the ownership guard: teardown cannot run
    // between the two writes, so a flush is either wholly before it — and
    // cleared by it — or wholly rejected.
    if !mono.is_empty() {
        let Ok(mut m) = sample_buf.try_lock() else {
            return false;
        };
        let room = MONO_CAP.saturating_sub(mono.len());
        if m.len() > room {
            let d = m.len() - room;
            m.drain(0..d);
        }
        m.extend_from_slice(mono);
    }
    if !pairs.is_empty() {
        let room = STEREO_CAP.saturating_sub(pairs.len());
        if v.frames.len() > room {
            let d = v.frames.len() - room;
            v.frames.drain(0..d);
        }
        v.frames.extend_from_slice(pairs);
    }
    true
}

/// Claim the buffer for a new stream of `channels` channels, and return the
/// generation the caller must present when writing.
///
/// Blocking, so it is for setup threads. A source that is appended ahead of
/// time must use [`try_claim_stereo_stream`] at its first pull instead — see
/// there for why claiming at construction is wrong.
pub fn begin_stereo_stream(buf: &StereoBuf, channels: u16) -> u64 {
    let mut v = match buf.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    claim_locked(&mut v, channels)
}

/// What happened when a deferred tap tried to take the buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StereoClaim {
    /// The lock was held. Nothing was changed; try again on the next sample.
    Busy,
    /// The tap now owns the buffer, and must present this generation to write.
    Claimed(u64),
    /// The route this tap belongs to is over. It must never try again — a
    /// retry would be a claim against whatever session is running now.
    Stale,
}

/// Claim the buffer without blocking and without allocating, for the route
/// `expected_epoch` names.
///
/// This exists for the shared gapless path. `append_next` hands the mixer the
/// next track up to two seconds before it is audible, so a source that claimed
/// at construction would take the display away from the track still playing.
/// The claim therefore happens at the first pull — the moment the mixer
/// actually reads the source, which is the moment it becomes audible — and that
/// happens on the audio thread.
///
/// The epoch is checked *under the same lock* that would perform the claim, so
/// there is no window between deciding the route is still current and acting on
/// it. Checking it beforehand and claiming afterwards would be exactly the race
/// this is here to close.
///
/// No allocation (capacity was reserved at construction, and `clear` never
/// allocates) and no blocking (a busy lock simply means trying again on the
/// next pull, a fraction of a millisecond later).
pub fn try_claim_stereo_stream(buf: &StereoBuf, channels: u16, expected_epoch: u64) -> StereoClaim {
    let Ok(mut v) = buf.try_lock() else {
        return StereoClaim::Busy;
    };
    if v.route_epoch != expected_epoch {
        return StereoClaim::Stale;
    }
    StereoClaim::Claimed(claim_locked(&mut v, channels))
}

fn claim_locked(v: &mut StereoTapBuf, channels: u16) -> u64 {
    v.frames.clear();
    v.channels = channels;
    v.generation = v.generation.wrapping_add(1);
    v.generation
}

/// End the current playback session as far as the analyser is concerned.
///
/// One authoritative operation, performed under one lock: the route epoch
/// advances, the frames and the channel claim go, and the write generation
/// advances too. After it, no tap built for the old route can claim, and no tap
/// that had already claimed can write.
///
/// Unconditional, so it is only for a caller that is tearing a route down —
/// releasing the shared sink, stopping, halting, a failed open, or a native DSD
/// route starting with no PCM tap at all. Anything releasing one owner among
/// several must use [`end_stereo_stream_if_generation`], which cannot touch the
/// epoch.
pub fn invalidate_live_pcm_route(stereo_buf: &StereoBuf, sample_buf: &SampleBuf) {
    let mut v = match stereo_buf.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    v.route_epoch = v.route_epoch.wrapping_add(1);
    retire_locked(&mut v);
    // Under the same guard, in the same order publication uses. Clearing the
    // Mix mattered more than clearing the pairs and was the half that was
    // missing: a stopped player, a failed open, a DoP or native route — all of
    // them left the previous track's mono PCM in place, and the bars went on
    // drawing it.
    if let Ok(mut m) = sample_buf.lock() {
        m.clear();
    }
}

/// Retire the lease only if `generation` is still the live one.
///
/// This is what every tap uses when it is dropped, and it is the whole reason a
/// generation exists. A track being torn down and its successor starting are
/// not ordered against each other: on the shared route the outgoing source is
/// dropped by the mixer at the same rollover that the incoming one is first
/// pulled, and on the bit-perfect route one tap deliberately outlives several
/// tracks. An unconditional retire from either would blank a display that
/// something else is legitimately feeding.
///
/// `try_lock`, because a source can be dropped on the audio thread. A busy lock
/// costs nothing: the successor's claim supersedes the stale one anyway, and
/// the engine's own teardown is unconditional.
///
/// It deliberately does **not** advance the route epoch. A source being dropped
/// is one owner going away, not the session ending; on the gapless path the
/// outgoing source is dropped while its successor — which belongs to the same
/// session — is already playing, and ending the session there would forbid that
/// successor from ever claiming.
pub fn end_stereo_stream_if_generation(buf: &StereoBuf, generation: u64) -> bool {
    let Ok(mut v) = buf.try_lock() else { return false };
    if v.generation != generation {
        return false;
    }
    retire_locked(&mut v);
    true
}

fn retire_locked(v: &mut StereoTapBuf) {
    v.frames.clear();
    v.channels = channels::NO_LIVE_TAP;
    v.generation = v.generation.wrapping_add(1);
}

/// Left in an overlay. Cyan and magenta are near-complements at similar
/// luminance, so neither reads as "the important one", and both hold their
/// identity over any album art the plot happens to be sitting on — which the
/// palette accent, being derived from that art, would not.
pub const CHANNEL_L_COLOR: Color32 = Color32::from_rgb(64, 208, 226);
/// Right in an overlay.
pub const CHANNEL_R_COLOR: Color32 = Color32::from_rgb(232, 92, 196);

/// One frame of spectra for the renderer.
///
/// Deliberately independent of any cache encoding: the future channel-aware
/// cache will fill the same shape from a different source, and nothing here
/// encodes L/R as a doubled bar count or a second file.
pub struct ChannelFrame<'a> {
    pub mix: &'a [f32],
    /// `None` means there is no left channel to draw — never a copy of `mix`.
    pub left: Option<&'a [f32]>,
    pub right: Option<&'a [f32]>,
}

impl ChannelFrame<'_> {
    /// Whether there is nothing on the plot.
    ///
    /// Every series present has to be silent, not just the mix: in a channel
    /// view the mix is not what is drawn, and testing it alone put the
    /// "Analyzing…" overlay on top of a perfectly live left/right plot.
    pub fn is_silent(&self) -> bool {
        let quiet = |s: &[f32]| s.iter().all(|&m| m <= 0.001);
        quiet(self.mix)
            && self.left.is_none_or(quiet)
            && self.right.is_none_or(quiet)
    }

    /// Both channels, or nothing. `Split` and `Overlay` need the pair.
    pub fn pair(&self) -> Option<(&[f32], &[f32])> {
        match (self.left, self.right) {
            (Some(l), Some(r)) => Some((l, r)),
            _ => None,
        }
    }
}

/// Full per-track analysis computed in the background waveform thread.
#[derive(Clone, Debug)]
pub struct TrackAnalysis {
    pub integrated_lufs: f32,      // f32::NEG_INFINITY if silence
    pub dr_score: u32,             // 0–20 integer DR score
    pub peak_dbfs: f32,            // max sample peak in dBFS
    pub clip_count: u32,           // samples at/near 0 dBFS
    pub clip_positions: Vec<f32>,  // normalised 0..1 positions (capped at 500)
    pub bpm: f32,                  // estimated tempo (0 = undetermined)
    pub key_name: String,          // e.g. "A minor", "C# major"
    pub loudness_history: Vec<f32>,// per-second integrated LUFS
}

// ---------------------------------------------------------------------------
// SpectrumSource — thin rodio Source wrapper that taps samples into SampleBuf
// ---------------------------------------------------------------------------

/// Samples are batched on the audio thread and flushed in bulk every
/// BATCH_SIZE mono frames — reduces mutex lock attempts from ~88k/sec
/// (per-sample) down to ~172/sec, cutting audio-thread overhead 512×.
const BATCH_SIZE: usize = 512;

/// Where a deferred tap is in the claim protocol.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TapClaim {
    /// Has not claimed yet. Tries again on every pull.
    Pending,
    /// Owns the buffer, and presents this generation to write.
    Owned(u64),
    /// Belongs to a route that has ended. Never tries again, and gathers
    /// nothing.
    Disabled,
}

pub struct SpectrumSource<S>
where
    S: rodio::Source + Send + 'static,
    S::Item: SampleToF32 + rodio::Sample,
{
    inner: S,
    buf: SampleBuf,
    stereo_buf: StereoBuf,
    /// Where this source is in the claim protocol.
    ///
    /// A source is constructed when it is appended, which on the gapless path
    /// is up to two seconds before anyone hears it. Claiming then would take
    /// the display away from the track still playing, so the claim waits for
    /// the first pull — and by then the session it belongs to may be over.
    claim: TapClaim,
    /// Route epoch captured at construction. The claim is only valid within it.
    route_epoch: u64,
    channels: u16,
    ch_idx: u16,
    pending_l: f32,
    frame_sum: f32,
    sample_batch: Vec<f32>,
    stereo_batch: Vec<[f32; 2]>,
}

impl<S> SpectrumSource<S>
where
    S: rodio::Source + Send + 'static,
    S::Item: SampleToF32 + rodio::Sample,
{
    pub fn new(inner: S, buf: SampleBuf, stereo_buf: StereoBuf) -> Self {
        let channels = inner.channels();
        // Reserve both buffers now, on the thread assembling the rodio chain,
        // so no flush and no claim can ever allocate — and take the session
        // this source belongs to at the same time, because by the first pull
        // the engine may have moved on and the whole point is to be able to
        // tell. Do *not* claim now: on the gapless path this source is appended
        // while the previous track is still playing, and taking the display
        // from it two seconds early is exactly the bug this ordering avoids.
        let route_epoch = prepare_live_tap(&stereo_buf, &buf);
        Self {
            inner, buf, stereo_buf, claim: TapClaim::Pending, route_epoch, channels,
            ch_idx: 0, pending_l: 0.0, frame_sum: 0.0,
            // Both batches fill at one entry per *frame* and are flushed on the
            // mono batch reaching BATCH_SIZE, so the stereo batch reaches
            // BATCH_SIZE too. It was built with room for half that and grew on
            // its first flush — on the thread feeding the device.
            sample_batch: Vec::with_capacity(BATCH_SIZE),
            stereo_batch: Vec::with_capacity(BATCH_SIZE),
        }
    }
}

impl<S> Iterator for SpectrumSource<S>
where
    S: rodio::Source + Send + 'static,
    S::Item: SampleToF32 + rodio::Sample,
{
    type Item = S::Item;

    fn next(&mut self) -> Option<S::Item> {
        let s = self.inner.next()?;
        let f = s.to_spectrum_f32();

        // The first sample anyone pulls from this source is the moment it
        // becomes the thing being played, which is the only honest moment to
        // take the display. Neither allocating nor blocking — see
        // `try_claim_stereo_stream`.
        //
        // `Busy` leaves the state Pending, so the next sample tries again.
        // `Stale` is final: the session this source was built for is over, and
        // the mixer is only in here because `Sink::stop` is an atomic the
        // callback notices asynchronously. Retrying would eventually succeed
        // against whatever route is running now, which is the failure this
        // whole protocol exists to prevent.
        if matches!(self.claim, TapClaim::Pending) {
            let attempt =
                try_claim_stereo_stream(&self.stereo_buf, self.channels, self.route_epoch);
            self.claim = match attempt {
                StereoClaim::Busy => TapClaim::Pending,
                StereoClaim::Claimed(g) => TapClaim::Owned(g),
                StereoClaim::Stale => TapClaim::Disabled,
            };
        }

        // A source whose route is over has nowhere to put anything, so it
        // gathers nothing — neither pairs nor the mono average below. It used
        // to keep feeding the Mix, which is the series most people watch.
        if matches!(self.claim, TapClaim::Disabled) {
            return Some(s);
        }

        // Accumulate stereo pairs locally.
        if self.channels == 2 {
            if self.ch_idx == 0 {
                self.pending_l = f;
            } else {
                self.stereo_batch.push([self.pending_l, f]);
            }
        }
        // The analyser reads this buffer as one mono stream sampled at the
        // track's rate. Pushing interleaved channels made it a 2× (or N×)
        // sample-and-hold instead: every partial appeared at f/N with a mirror
        // image at Nyquist − f/N, which is why bass showed up again at the top
        // of the display. Average each frame down, exactly as the pre-process
        // decoder does.
        let ch = self.channels.max(1);
        self.frame_sum += f;
        self.ch_idx += 1;
        if self.ch_idx >= ch {
            self.sample_batch.push(self.frame_sum / ch as f32);
            self.frame_sum = 0.0;
            self.ch_idx = 0;
        }

        // Flush both buffers together, under one lease, once per batch.
        //
        // Only an owner publishes. A source still waiting to claim is not the
        // live one, and one whose route ended never will be; both used to feed
        // the mono buffer regardless, because mono was written on its own before
        // anything checked. The batches are cleared either way — holding them
        // back for a later flush would show the display audio from the wrong
        // side of a discontinuity.
        if self.sample_batch.len() >= BATCH_SIZE {
            if let TapClaim::Owned(lease) = self.claim {
                publish_live_pcm(
                    &self.stereo_buf,
                    &self.buf,
                    lease,
                    &self.sample_batch,
                    &self.stereo_batch,
                );
            }
            self.sample_batch.clear();
            self.stereo_batch.clear();
        }
        Some(s)
    }
}

/// Give the lease back when the source is dropped — which is exactly when the
/// mixer is finished with this track, whether that is a rollover, a stop, or a
/// route being torn down under it.
///
/// Generation-qualified, so a source being dropped at a gapless rollover cannot
/// blank the successor that has already claimed. A source that never claimed —
/// a queued track the user skipped past — has nothing to give back.
impl<S> Drop for SpectrumSource<S>
where
    S: rodio::Source + Send + 'static,
    S::Item: SampleToF32 + rodio::Sample,
{
    fn drop(&mut self) {
        // Generation-qualified, and deliberately never epoch-touching: this is
        // one owner going away, not the session ending. Ending the session here
        // would forbid a queued successor — which shares this epoch — from ever
        // claiming.
        if let TapClaim::Owned(lease) = self.claim {
            end_stereo_stream_if_generation(&self.stereo_buf, lease);
        }
    }
}

impl<S> rodio::Source for SpectrumSource<S>
where
    S: rodio::Source + Send + 'static,
    S::Item: SampleToF32 + rodio::Sample,
{
    fn current_frame_len(&self) -> Option<usize> { self.inner.current_frame_len() }
    fn channels(&self) -> u16 { self.inner.channels() }
    fn sample_rate(&self) -> u32 { self.inner.sample_rate() }
    fn total_duration(&self) -> Option<Duration> { self.inner.total_duration() }

    /// Delegate seek to the inner decoder so `Sink::try_seek` actually works.
    /// Without this the default impl returns `Err(NotSupported)` and audio
    /// never jumps, even though rodio/symphonia fully supports seeking.
    fn try_seek(&mut self, pos: Duration) -> Result<(), rodio::source::SeekError> {
        self.inner.try_seek(pos)
    }
}

// ---------------------------------------------------------------------------
// Background pre-processing messages
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub enum PreMessage {
    Done {
        frames: Vec<Vec<f32>>,
        frame_rate: f64,
        waveform: Vec<f32>,
        analysis: TrackAnalysis,
    },
    Error(String),
    /// The user pressed Abort. Distinct from `Error` so the UI can drop back to
    /// the live view quietly instead of reporting a failure, and so nothing is
    /// written to the cache.
    Aborted,
}

// ---------------------------------------------------------------------------
// SpectrumAnalyzer — owns FFT state, waterfall history, pre-process frames
// ---------------------------------------------------------------------------

pub struct SpectrumAnalyzer {
    pub sample_buf: SampleBuf,
    window_coeffs: Vec<f32>,
    fft_plan: Arc<dyn rustfft::Fft<f32>>,
    /// Lightweight plan for real-time display: always fft_size × 2.
    /// The full fft_plan (fft_size × pad_factor) is used only for pre-process
    /// (runs in a background thread). Keeping real-time at 2× ensures the
    /// UI thread never spends more than ~0.5 ms on an FFT frame.
    rt_fft_plan: Arc<dyn rustfft::Fft<f32>>,
    pub magnitudes: Vec<f32>,
    smoothed: Vec<f32>,
    /// What peak-hold should consider this tick: the largest displayed value
    /// across every source row crossed since the last one.
    ///
    /// Separate from `smoothed` because a UI tick can span several cached rows,
    /// and reading only the last of them drops a one-row transient on the
    /// floor — the peak marker exists precisely to catch those.
    pub peak_input: Vec<f32>,
    /// Last pre-processed row consumed, so a row is filtered exactly once and
    /// a repeated tick within the same row is a no-op. `None` means there is no
    /// filter state to continue from and the next tick must snap.
    last_pre_frame: Option<usize>,
    /// How many rows the ring keeps. Derived from the wanted span and the
    /// producer's rate whenever either changes.
    pub waterfall_rows: usize,
    /// Cached rows consumed since the debug overlay last read the counter.
    rows_consumed: u64,
    /// Bumped whenever the cached matrix is installed or dropped.
    ///
    /// The window owns presentation state the analyser cannot see — peak decay,
    /// the uploaded waterfall texture — and a cache can arrive from a worker at
    /// any moment, including while paused, when nothing is calling `tick_pre`.
    /// A counter is what lets the window notice, without the analyser having to
    /// know what a texture is.
    pre_revision: u64,
    /// The retention the most recent real-time frame used, published for the
    /// consumers that run after `process_realtime` returns.
    frame_alpha: Option<f32>,
    /// The next successfully computed real-time frame is a snap.
    ///
    /// Zeroing the smoother is not enough on its own: the first frame after a
    /// discontinuity still applies whatever alpha the elapsed time implies, so
    /// at a high retention the display fades up from silence over a second or
    /// more instead of showing the audio that is playing. Set on every producer
    /// change, consumed only when a transform actually happens — an early
    /// return for want of PCM must not spend it.
    snap_next_realtime: bool,
    /// Displayed left/right spectra. Empty whenever no channel view is asking
    /// for them, which is how the renderer knows there is nothing to draw
    /// without consulting the availability logic a second time.
    pub bars_left: Vec<f32>,
    pub bars_right: Vec<f32>,
    smoothed_left: Vec<f32>,
    smoothed_right: Vec<f32>,
    /// Windowed, zero-padded scratch for one channel transform. Half a megabyte
    /// at the largest window, so it is kept rather than allocated per frame.
    ch_scratch: Vec<Complex<f32>>,
    /// Channel transforms run since the counter was last read. Exists so a test
    /// can show that Mix really does not pay for a second FFT, rather than
    /// asserting it from the shape of the code.
    channel_ffts: u64,
    pub sample_rate: u32,
    pub bar_count: usize,
    // Configurable FFT parameters
    pub fft_size: usize,
    pub window_fn: WindowFn,
    /// Real-time retention, applied once per accepted analyser tick.
    ///
    /// The recurrence this release ships is the one v1.4.5 shipped, unchanged:
    /// `new = old*s + frame*(1-s)`, once per tick that the `max_fps` throttle
    /// lets through. Normalising it to a reference rate — which the previous
    /// phase did — changed the response of a control that was working, on top
    /// of fixing one that was not, and cost real-time about three times its
    /// speed on a fast display.
    pub smoothing: f32, // 0.0 = off, 0.9 = very smooth
    /// Pre-process retention, at the 60 Hz reference — a different quantity
    /// from `smoothing`, in different units, with its own persisted value.
    ///
    /// Pre-process consumes cached rows in source time, so a per-tick retention
    /// has no meaning there: the number of rows in a tick depends on the
    /// display. This one is converted for the interval actually crossed, which
    /// is what makes the result the same on every machine.
    ///
    /// See [`timing::DEFAULT_PRE_SMOOTHING`] for why it does not default to
    /// `smoothing`'s 0.75.
    pub pre_smoothing: f32,
    pub min_freq: f32,
    pub max_freq: f32,
    /// Optional per-bar dB correction (ISO 226).  Empty = disabled.
    pub eq_weights: Vec<f32>,
    /// Sub-bin interpolation strategy.
    pub interp_mode: InterpolationMode,
    /// Zero-padding multiplier applied before the FFT (1 = off, 2/4/… = denser bins).
    pub pad_factor: usize,
    /// Analysis window overlap (0.5 = 50%, 0.75 = 75%, 0.875 = 87.5%). Affects frame rate.
    pub overlap: f32,
    /// Multi-bin bar weighting strategy.
    pub bar_mapping: BarMappingMode,
    /// Target PCM rate DSD tracks are decimated to for analysis (176.4 kHz
    /// default; 352.8 kHz doubles the frame rate for a faster-reacting
    /// spectrum). Ignored for PCM files.
    pub dsd_rate: u32,
    /// Superlet parameters, live only when `bar_mapping == Superlet`.
    pub aslt_cfg: aslt::AsltConfig,
    /// Which rung of the quality ladder `aslt_cfg` came from; `None` = Custom,
    /// i.e. the user is driving `o_min`/`o_max` by hand.
    pub aslt_preset: Option<aslt::AsltPreset>,
    /// Pre-process frame rate, in frames per second.
    ///
    /// The FFT path derives its frame rate from window length and overlap; the
    /// superlet has no window length to derive it from, so it is stated
    /// directly. Defining it in *time* rather than in samples also keeps cost
    /// linear in sample rate — with a sample-denominated hop, a 176.4 kHz DSD
    /// analysis would cost 16× a 44.1 kHz one instead of 4×.
    pub pre_fps: f32,
    /// Set to abort a running analysis. The worker checks it between bars and
    /// returns [`PreMessage::Aborted`] without writing a cache.
    pub abort_analysis: Arc<AtomicBool>,
    /// Seconds remaining in the running analysis; `usize::MAX` = not yet known.
    pub eta_secs: Arc<AtomicUsize>,
    /// Track the in-flight analysis belongs to, so a repeat of the same file can
    /// be recognised and left alone rather than restarted.
    pub analyzing_path: Option<PathBuf>,
    /// Ceiling on the whole cache directory, in gigabytes. 0 disables eviction.
    pub cache_budget_gb: f32,

    // Pre-process pending results
    pub pending_waveform: Option<Vec<f32>>,
    pub pending_analysis: Option<TrackAnalysis>,

    // Pre-process
    pub pre_frames: Vec<Vec<f32>>,
    pub pre_frame_rate: f64,
    pub pre_receiver: Option<std::sync::mpsc::Receiver<PreMessage>>,
    /// True while a background analysis thread is running. Arc so the thread can clear it.
    pub is_analyzing: Arc<AtomicBool>,
    /// 0–100 progress updated by the background thread.
    pub analysis_progress: Arc<AtomicUsize>,

    // Waterfall
    pub waterfall: Vec<Vec<f32>>,
    pub waterfall_dirty: bool,
    /// Total rows ever pushed. The renderer diffs this against what it has
    /// already uploaded to know how many rows are genuinely new.
    pub waterfall_seq: u64,
    /// When false, push_waterfall() is a no-op (used when the waterfall viz is not active).
    pub waterfall_enabled: bool,
    /// Raw FFT bin magnitudes (half-spectrum) from the most recent real-time frame.
    /// Used by the spectrogram and octave-band RTA views.
    pub last_fft_norms: Vec<f32>,
}

impl SpectrumAnalyzer {
    pub fn new(sample_buf: SampleBuf) -> Self {
        let fft_size = DEFAULT_FFT_SIZE;
        const DEFAULT_PAD: usize = 16;
        let mut planner = FftPlanner::new();
        let fft_plan = planner.plan_fft_forward(fft_size * DEFAULT_PAD);
        let rt_fft_plan = planner.plan_fft_forward(fft_size * 2);
        let window_coeffs = make_window(fft_size, &WindowFn::Hann);
        Self {
            sample_buf,
            window_coeffs,
            fft_plan,
            rt_fft_plan,
            magnitudes: vec![0.0f32; DEFAULT_BAR_COUNT],
            smoothed: vec![0.0f32; DEFAULT_BAR_COUNT],
            peak_input: vec![0.0f32; DEFAULT_BAR_COUNT],
            last_pre_frame: None,
            waterfall_rows: WATERFALL_ROWS_FALLBACK,
            rows_consumed: 0,
            pre_revision: 0,
            frame_alpha: None,
            snap_next_realtime: true,
            bars_left: Vec::new(),
            bars_right: Vec::new(),
            smoothed_left: Vec::new(),
            smoothed_right: Vec::new(),
            ch_scratch: Vec::new(),
            channel_ffts: 0,
            sample_rate: 44_100,
            bar_count: DEFAULT_BAR_COUNT,
            fft_size,
            window_fn: WindowFn::Hann,
            smoothing: 0.75,
            pre_smoothing: timing::DEFAULT_PRE_SMOOTHING,
            min_freq: DEFAULT_MIN_FREQ,
            max_freq: DEFAULT_MAX_FREQ,
            eq_weights: Vec::new(),
            interp_mode: InterpolationMode::None,
            pad_factor: DEFAULT_PAD,
            overlap: 0.875,
            bar_mapping: BarMappingMode::Cqt,
            dsd_rate: crate::dsd::decimate::DEFAULT_ANALYSIS_RATE,
            aslt_cfg: aslt::AsltPreset::Standard.config(),
            aslt_preset: Some(aslt::AsltPreset::Standard),
            pre_fps: DEFAULT_PRE_FPS,
            abort_analysis: Arc::new(AtomicBool::new(false)),
            eta_secs: Arc::new(AtomicUsize::new(usize::MAX)),
            analyzing_path: None,
            cache_budget_gb: DEFAULT_CACHE_BUDGET_GB,
            pending_waveform: None,
            pending_analysis: None,
            pre_frames: Vec::new(),
            pre_frame_rate: 60.0,
            pre_receiver: None,
            is_analyzing: Arc::new(AtomicBool::new(false)),
            analysis_progress: Arc::new(AtomicUsize::new(0)),
            waterfall: Vec::new(),
            waterfall_dirty: false,
            waterfall_seq: 0,
            waterfall_enabled: false,
            last_fft_norms: Vec::new(),
        }
    }

    /// Rebuild the FFT plan and window coefficients after parameter changes.
    pub fn rebuild_fft(&mut self) {
        let mut planner = FftPlanner::new();
        self.fft_plan = planner.plan_fft_forward(self.fft_size * self.pad_factor);
        self.rt_fft_plan = planner.plan_fft_forward(self.fft_size * 2);
        self.window_coeffs = make_window(self.fft_size, &self.window_fn);
        // A different window length means a different number of bins behind
        // each bar, so the channel filter state no longer describes the same
        // measurement. The scratch is dropped with it rather than resized.
        self.reset_channels();
        self.ch_scratch = Vec::new();
    }


    fn bins_to_bars(&self, fft_out: &[Complex<f32>], sr: u32, n_bars: usize, padded_size: usize) -> Vec<f32> {
        let fft_size = self.fft_size;
        let half = padded_size / 2;
        let log_min = self.min_freq.log10();
        let log_max = (sr as f32 / 2.0).min(self.max_freq).log10();
        let norms: Vec<f32> = fft_out[..half].iter().map(|c| c.norm()).collect();
        let scale = fft_size as f32; // normalize by window length, not padded length

        // Pre-compute Q factor for CQT mode: Q = 1 / (2^(1/B) - 1)
        // where B = bars per octave = n_bars / log2(fmax/fmin)
        let n_octaves = (log_max - log_min) / 2_f32.log10();
        let bins_per_oct = n_bars as f32 / n_octaves.max(0.1);
        let cqt_q = 1.0 / (2.0_f32.powf(1.0 / bins_per_oct) - 1.0);

        (0..n_bars)
            .map(|bar| {
                // Bar edges follow whichever scale the display is on, or the
                // FFT bins would be gathered into bars that sit somewhere else.
                let fscale = self.aslt_cfg.scale;
                let (lo_hz, hi_hz) = (self.min_freq, self.max_freq);
                let mag = if self.bar_mapping == BarMappingMode::Cqt {
                    // CQT: centre-frequency Hann kernel with constant-Q bandwidth
                    let tc  = (bar as f32 + 0.5) / n_bars as f32;
                    let f_c = fscale.freq_at(tc, lo_hz, hi_hz);
                    let bc  = (f_c * padded_size as f32 / sr as f32).clamp(1.0, half as f32 - 1.0);
                    cqt_kernel(&norms, bc, cqt_q, half)
                } else {
                    let t0 = bar as f32 / n_bars as f32;
                    let t1 = (bar + 1) as f32 / n_bars as f32;
                    let freq_lo = fscale.freq_at(t0, lo_hz, hi_hz);
                    let freq_hi = fscale.freq_at(t1, lo_hz, hi_hz);
                    // Both ends are bounded by the spectrum that exists.
                    //
                    // `max_freq` defaults to 24 kHz, which is above Nyquist for
                    // a 44.1 kHz track, so the top bars ask for bins the FFT
                    // never produced. Only `fbin_hi` used to be clamped; a bar
                    // whose *low* edge was already past the end then produced a
                    // negative width, took the sub-bin branch, and indexed the
                    // midpoint of a range that started off the end of the array.
                    // Those bars read the topmost bin instead, which is what the
                    // constant-Q branch has always done with them.
                    let top = half as f32 - 1.0;
                    let fbin_lo = (freq_lo * padded_size as f32 / sr as f32).clamp(1.0, top);
                    let fbin_hi = (freq_hi * padded_size as f32 / sr as f32)
                        .clamp(fbin_lo, top);

                    // Sub-bin: interpolated; Multi-bin: weighted overlap average.
                    if fbin_hi - fbin_lo <= 1.0 {
                        let center = (fbin_lo + fbin_hi) * 0.5;
                        interp_sub_bin(&norms, center, &self.interp_mode)
                    } else {
                        let b_start = fbin_lo.floor() as usize;
                        let b_end   = (fbin_hi.ceil() as usize).min(half - 1);
                        let bc      = (fbin_lo + fbin_hi) * 0.5;
                        let sigma   = ((fbin_hi - fbin_lo) * 0.5).max(0.5);
                        let mut wsum = 0.0_f32;
                        let mut weight = 0.0_f32;
                        for (b_idx, &norm_b) in norms[b_start..=b_end].iter().enumerate() {
                            let b = b_start + b_idx;
                            let w = match &self.bar_mapping {
                                BarMappingMode::FlatOverlap => {
                                    (fbin_hi.min(b as f32 + 1.0) - fbin_lo.max(b as f32)).max(0.0)
                                }
                                BarMappingMode::Gaussian
                                | BarMappingMode::Cqt
                                | BarMappingMode::Superlet => {
                                    let center_b = b as f32 + 0.5;
                                    (-(center_b - bc).powi(2) / (2.0 * sigma * sigma)).exp()
                                }
                            };
                            wsum += norm_b * w; weight += w;
                        }
                        if weight > 0.0 { wsum / weight } else { 0.0 }
                    }
                };

                let raw_db = 20.0 * (mag / scale).log10().max(-80.0);
                let corrected_db = if let Some(&w) = self.eq_weights.get(bar) { raw_db + w } else { raw_db };
                ((corrected_db + 80.0) / 80.0).clamp(0.0, 1.0)
            })
            .collect()
    }

    /// Run one live FFT frame and fold it into the display.
    ///
    /// `dt` is the wall time since the *previous accepted* analyser tick, not
    /// since the previous repaint: this is called from behind the `max_fps`
    /// throttle, and the smoothing contract is defined per unit of elapsed
    /// time. The first tick of a stream passes a very large `dt`, which
    /// correctly resolves to no smoothing at all.
    pub fn process_realtime(&mut self, dt: f64) {
        let fft_size = self.fft_size;
        // Real-time FFT uses 2× padding only — keeps the UI thread's per-frame
        // work at ~16 K points instead of 131 K (pad=16), eliminating the CPU
        // spike that caused audio buffer underruns when the spectrum was open.
        // The high-quality padded plan is used only by the background pre-process.
        let rt_padded = fft_size * 2;

        let samples: Vec<f32> = {
            let buf = match self.sample_buf.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            if buf.len() < fft_size {
                return;
            }
            let start = buf.len().saturating_sub(fft_size);
            buf[start..].to_vec()
        };

        let zero_pad = rt_padded - fft_size;
        let mut buf: Vec<Complex<f32>> = samples
            .iter()
            .zip(self.window_coeffs.iter())
            .map(|(s, w)| Complex { re: s * w, im: 0.0 })
            .chain(std::iter::repeat_n(Complex { re: 0.0, im: 0.0 }, zero_pad))
            .collect();
        self.rt_fft_plan.process(&mut buf);

        let half = rt_padded / 2;
        self.last_fft_norms = buf[..half].iter().map(|c| c.norm()).collect();
        let sr = self.sample_rate;
        let n = self.bar_count;
        let new_bars = self.bins_to_bars(&buf, sr, n, rt_padded);
        if self.smoothed.len() != new_bars.len() {
            self.smoothed = vec![0.0; new_bars.len()];
        }
        // Decided once, here, after a transform has actually happened — the
        // early return above must not spend the snap on a frame that was never
        // computed. The same decision is then handed to every other consumer of
        // this frame, so Left/Right and the octave meters snap with the Mix
        // instead of fading up from zero behind it.
        let alpha = self.take_realtime_alpha();
        for (s, m) in self.smoothed.iter_mut().zip(new_bars.iter()) {
            *s = *s * alpha + m * (1.0 - alpha);
        }
        self.magnitudes.clear();
        self.magnitudes.extend_from_slice(&self.smoothed);
        // One frame in, one value out: the interval peak-hold sees is just the
        // frame. The pre-process path is the one that can cross several.
        self.peak_input.clear();
        self.peak_input.extend_from_slice(&self.smoothed);
        // Published for this frame's other consumers, which run after this
        // returns and must not each re-derive it.
        self.frame_alpha = Some(alpha);
        // One row per accepted analyser tick — the producer's own rate, which
        // is what this has always been except for one release that capped it.
        let _ = dt;
        self.push_waterfall_row();
    }

    /// Run the left and right transforms and fold them into their own
    /// smoothers.
    ///
    /// Called only when a channel view is both selected and available. `Mix`
    /// must not pay for this: the mono path already produced everything it
    /// needs, and a second and third FFT per frame on the UI thread is exactly
    /// the cost that made the spectrum window a source of audio underruns
    /// before the real-time path was cut to 2x padding.
    ///
    /// `left` and `right` are the caller's copies of the shared tap, taken
    /// under one lock. They are equal in length by construction — the tap only
    /// ever pushes complete pairs.
    pub fn process_channels(&mut self, left: &[f32], right: &[f32], dt: f64) {
        let fft_size = self.fft_size;
        if left.len() < fft_size || right.len() < fft_size {
            // Not enough history yet. Leaving the previous bars in place would
            // freeze a stale spectrum on screen, so clear instead: the plot
            // shows nothing until there is something to show.
            self.bars_left.clear();
            self.bars_right.clear();
            return;
        }
        let rt_padded = fft_size * 2;
        if self.ch_scratch.len() != rt_padded {
            self.ch_scratch = vec![Complex { re: 0.0, im: 0.0 }; rt_padded];
        }
        let n = self.bar_count;
        let sr = self.sample_rate;
        // The Mix's decision for this frame, not a second one derived here: a
        // snap has to be a snap on every series or the plot shows a left and
        // right fading up behind a Mix that is already correct. `dt` is kept in
        // the signature for the fallback and for callers that drive this
        // directly.
        let alpha = self
            .frame_alpha
            .unwrap_or_else(|| self.smoothing.clamp(0.0, 1.0));
        let _ = dt;

        for side in 0..2 {
            let pcm = if side == 0 { left } else { right };
            let start = pcm.len() - fft_size;
            for (slot, (x, w)) in self.ch_scratch[..fft_size]
                .iter_mut()
                .zip(pcm[start..].iter().zip(self.window_coeffs.iter()))
            {
                *slot = Complex { re: x * w, im: 0.0 };
            }
            for slot in self.ch_scratch[fft_size..].iter_mut() {
                *slot = Complex { re: 0.0, im: 0.0 };
            }
            self.rt_fft_plan.process(&mut self.ch_scratch);
            self.channel_ffts = self.channel_ffts.saturating_add(1);
            // The same mapping the mono path uses, so the three spectra share
            // one frequency axis and one dB scale and cannot drift apart.
            let bars = self.bins_to_bars(&self.ch_scratch, sr, n, rt_padded);

            let (smoothed, out) = if side == 0 {
                (&mut self.smoothed_left, &mut self.bars_left)
            } else {
                (&mut self.smoothed_right, &mut self.bars_right)
            };
            if smoothed.len() != bars.len() {
                *smoothed = vec![0.0; bars.len()];
            }
            for (sm, b) in smoothed.iter_mut().zip(bars.iter()) {
                *sm = *sm * alpha + b * (1.0 - alpha);
            }
            out.clear();
            out.extend_from_slice(smoothed);
        }
    }

    /// The retention this real-time frame is folded in with, taken once.
    ///
    /// Real-time is a per-accepted-tick recurrence — the value is used as it is
    /// rather than converted for an interval, which is what v1.4.5 did and what
    /// the owner's ears are calibrated to. The snap is consumed here so that
    /// every consumer of the frame sees the same answer: previously only the
    /// Mix asked, and the channel and octave series each applied ordinary
    /// smoothing, so after any reset the first Left/Right or octave frame faded
    /// up from silence underneath a Mix that had snapped.
    fn take_realtime_alpha(&mut self) -> f32 {
        if std::mem::take(&mut self.snap_next_realtime) {
            0.0
        } else {
            self.smoothing.clamp(0.0, 1.0)
        }
    }

    /// The decision the most recent real-time frame was folded in with, for the
    /// consumers that run after it. `None` before any frame has been computed.
    pub fn frame_alpha(&self) -> Option<f32> {
        self.frame_alpha
    }

    /// Forget the channel spectra and their filter state.
    ///
    /// Called whenever what they describe changes underneath them: a new track,
    /// a seek, a different FFT size or mapping, or the channel view becoming
    /// unavailable. Without it a paused-then-restarted stream would decay from
    /// the previous track's last frame.
    pub fn reset_channels(&mut self) {
        self.bars_left.clear();
        self.bars_right.clear();
        self.smoothed_left.clear();
        self.smoothed_right.clear();
    }

    /// Channel transforms run since this was last called.
    pub fn take_channel_ffts(&mut self) -> u64 {
        std::mem::take(&mut self.channel_ffts)
    }

    /// Return transforms to the counter when a sampling window was too short
    /// to divide by.
    pub fn add_channel_ffts(&mut self, n: u64) {
        self.channel_ffts = self.channel_ffts.saturating_add(n);
    }

    /// What the renderer should draw, with the channels attached only when they
    /// genuinely exist.
    ///
    /// `left`/`right` are `None` rather than a copy of `mix` when unavailable,
    /// so there is no shape in which the renderer can draw the same data twice
    /// and label it L and R.
    pub fn channel_frame(&self) -> ChannelFrame<'_> {
        ChannelFrame {
            mix: &self.magnitudes,
            left: (!self.bars_left.is_empty()).then_some(&self.bars_left[..]),
            right: (!self.bars_right.is_empty()).then_some(&self.bars_right[..]),
        }
    }

    pub fn tick_pre(&mut self, elapsed: f64, dt: f64) {
        // Poll for completed background work
        let mut ready: Option<(Vec<Vec<f32>>, f64)> = None;
        if let Some(ref rx) = self.pre_receiver && let Ok(msg) = rx.try_recv() {
            match msg {
                PreMessage::Done { frames, frame_rate, waveform, analysis } => {
                    // The worker has just written a cache file; keep the whole
                    // directory inside its budget now rather than letting it
                    // grow unbounded between sessions.
                    if self.cache_budget_gb > 0.0 {
                        let budget = (self.cache_budget_gb as f64 * 1e9) as u64;
                        let (n, freed) = evict_cache_to_budget(budget, None);
                        if n > 0 {
                            crate::mlog!("[cache] evicted {n} file(s), freed {:.1} MB",
                                      freed as f64 / 1e6);
                        }
                    }
                    ready = Some((frames, frame_rate));
                    self.pending_waveform = Some(waveform);
                    self.pending_analysis = Some(analysis);
                }
                PreMessage::Error(_) | PreMessage::Aborted => {
                    // is_analyzing already cleared by ClearOnDrop in thread
                }
            }
        }
        if let Some((frames, rate)) = ready {
            crate::mlog!("[analysis] {} frames received by the display", frames.len());
            self.set_pre_frames(frames, rate);
            self.pre_receiver = None;
            // Flag already cleared by the thread's ClearOnDrop guard
        }

        let Some(target) = timing::target_frame(
            elapsed, self.pre_frame_rate, self.pre_frames.len(),
        ) else {
            // No pre-processed frames yet (first analysis still running, or no
            // cache for the current settings) — fall back to the live FFT so
            // the display is not dead meanwhile. For DSD there is no live PCM
            // to tap (the stream is a DoP carrier, or raw DSD on a native
            // route) and this early-returns; the
            // plot overlay reports analysis progress instead.
            self.process_realtime(dt);
            return;
        };

        let n = self.pre_frames[target].len();
        // Guard: if bar_count changed mid-stream, resize the derived buffers.
        // A resize discards the filter state along with the old width, so the
        // cursor has to go with it, or the next tick would continue a cascade
        // into vectors that no longer correspond to it.
        if self.smoothed.len() != n {
            self.smoothed = vec![0.0; n];
            self.last_pre_frame = None;
        }
        if self.magnitudes.len() != n {
            self.magnitudes = vec![0.0; n];
        }
        if self.peak_input.len() != n {
            self.peak_input = vec![0.0; n];
        }

        match timing::plan(self.last_pre_frame, target, timing::MAX_CATCHUP_FRAMES) {
            timing::FramePlan::Idle => {
                // The clock has not left the row already consumed. Leaving the
                // magnitudes, the filter state and the waterfall untouched is
                // what makes a repeated tick idempotent — the old code ran
                // another EMA step here, so the same row was smoothed once per
                // repaint and the visible decay tracked the monitor.
            }
            timing::FramePlan::Snap { to } => {
                self.peak_input.fill(0.0);
                self.apply_pre_frame(to, 0.0);
                self.last_pre_frame = Some(to);
                        self.push_waterfall_row();
            }
            timing::FramePlan::Consume { first, last } => {
                let step = timing::source_dt(self.pre_frame_rate);
                let alpha = timing::alpha_for_dt(self.pre_smoothing, step);
                // Reset before the cascade, not after: peak-hold wants the
                // largest value over the whole interval, and the interval is
                // exactly the rows about to be consumed.
                self.peak_input.fill(0.0);
                let _ = step;
                for frame in first..=last {
                    self.apply_pre_frame(frame, alpha);
                    // Every row the cursor crosses, not a decimation of them.
                    // The bars see all of these; the history has to as well, or
                    // a transient shows in one and not the other.
                    self.push_waterfall_row();
                }
                self.last_pre_frame = Some(last);
            }
        }
    }

    /// Fold cached row `frame` into the display with retention `alpha`.
    ///
    /// Equal-loudness weighting is applied *here*, not baked into the cache. It
    /// used to be folded in during analysis, which made it inert:
    /// `loudness_mode` is not part of the cache key, so switching it changed
    /// nothing and re-analysing simply reloaded the same file. Whichever mode
    /// happened to be selected when a track was first analysed was the mode it
    /// kept, permanently. As a per-bar dB offset on an already-normalised value
    /// it costs one add, so there is no reason for it to touch the expensive
    /// path at all.
    ///
    /// `alpha == 0.0` is a snap: the displayed value becomes the weighted cache
    /// row exactly, with no residue of whatever came before. That is what the
    /// smoothing slider at zero has to mean, and what it did not mean while
    /// this path hard-coded 0.5.
    fn apply_pre_frame(&mut self, frame: usize, alpha: f32) {
        let Some(mags) = self.pre_frames.get(frame) else { return };
        self.rows_consumed = self.rows_consumed.saturating_add(1);
        let w = &self.eq_weights;
        for (bar, (sm, m)) in self.smoothed.iter_mut().zip(mags.iter()).enumerate() {
            let v = match w.get(bar) {
                Some(&db) => (m + db / 80.0).clamp(0.0, 1.0),
                None => *m,
            };
            *sm = *sm * alpha + v * (1.0 - alpha);
            // Max across the interval, so a transient living in a row the UI
            // never displayed still reaches the peak marker.
            let p = &mut self.peak_input[bar];
            if *sm > *p {
                *p = *sm;
            }
        }
        self.magnitudes.copy_from_slice(&self.smoothed);
    }

    /// Snap the pre-process display to `elapsed_secs`, discarding filter state.
    ///
    /// Used after a seek, where continuing the cascade would either run
    /// backwards or replay minutes of rows that were never played. A no-op when
    /// there is no cache, so the caller does not have to check.
    pub fn snap_pre_to(&mut self, elapsed_secs: f64) {
        let Some(target) =
            timing::target_frame(elapsed_secs, self.pre_frame_rate, self.pre_frames.len())
        else {
            return;
        };
        let n = self.pre_frames[target].len();
        if self.smoothed.len() != n {
            self.smoothed = vec![0.0; n];
        }
        if self.magnitudes.len() != n {
            self.magnitudes = vec![0.0; n];
        }
        if self.peak_input.len() != n {
            self.peak_input = vec![0.0; n];
        }
        self.peak_input.fill(0.0);
        self.apply_pre_frame(target, 0.0);
        self.last_pre_frame = Some(target);
    }

    /// Cached rows consumed since the counter was last read, and reset.
    ///
    /// Exists so the debug overlay can report the rate rows are actually
    /// consumed at, which is the number the smoothing fix is about — the
    /// analyser tick rate beside it says nothing about how much of the cache
    /// reaches the screen.
    pub fn take_rows_consumed(&mut self) -> u64 {
        std::mem::take(&mut self.rows_consumed)
    }

    /// Return rows to the counter when a sampling window was too short to
    /// divide by, so the reported rate averages rather than dropping them.
    pub fn add_rows_consumed(&mut self, n: u64) {
        self.rows_consumed = self.rows_consumed.saturating_add(n);
    }

    /// Emit one waterfall row from the current magnitudes.
    fn push_waterfall_row(&mut self) {
        // push_waterfall no-ops unless the waterfall view is active; checking
        // here as well avoids the clone.
        if self.waterfall_enabled {
            self.push_waterfall(self.magnitudes.clone());
        }
    }

    /// Hop between pre-processed frames, in samples. Superlet states its frame
    /// rate directly; the FFT path derives it from window length and overlap.
    pub fn pre_hop(&self) -> usize {
        if self.bar_mapping == BarMappingMode::Superlet {
            aslt::hop_for_fps(self.sample_rate, self.pre_fps)
        } else {
            ((self.fft_size as f32 * (1.0 - self.overlap)).round() as usize).max(1)
        }
    }

    /// Ask a running analysis to stop. Returns to the live view; no cache is
    /// written, so nothing partial is ever reloaded later as if it were whole.
    pub fn abort_preprocess(&self) {
        self.abort_analysis.store(true, Ordering::Relaxed);
    }

    pub fn start_preprocess(&mut self, path: PathBuf) {
        // Guard: refuse to spawn a second thread if one is already running.
        if self.is_analyzing.load(Ordering::Relaxed) {
            crate::mlog!(
                "[analysis] refused, one already running at {}%",
                self.analysis_progress.load(Ordering::Relaxed),
            );
            return;
        }
        let n_bars = self.bar_count;
        let cache = cache_path_for(
            &path, n_bars, self.fft_size, self.pad_factor, self.overlap,
            &self.window_fn, self.min_freq, self.max_freq, &self.bar_mapping, &self.interp_mode,
            self.dsd_rate, &self.aslt_cfg, self.pre_fps,
        );
        if let Some(frames) = load_cache(&cache, n_bars) {
            let rate = self.sample_rate as f64 / self.pre_hop() as f64;
            // Derive a waveform from cached frames using mean bar magnitude per frame.
            if !frames.is_empty() {
                let wf_n = 1000usize;
                let n_frames = frames.len();
                let raw: Vec<f32> = (0..wf_n).map(|i| {
                    let fi = (i * n_frames / wf_n).min(n_frames - 1);
                    let sum: f32 = frames[fi].iter().sum();
                    sum / frames[fi].len().max(1) as f32
                }).collect();
                let peak = raw.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
                self.pending_waveform = Some(raw.iter().map(|&v| v / peak).collect());
            }
            self.set_pre_frames(frames, rate);
            return;
        }
        self.clear_pre_frames();
        self.analysis_progress.store(0, Ordering::Relaxed);
        self.analyzing_path = Some(path.clone());
        // Fresh Arc rather than storing false: a previous aborted run may still
        // be unwinding and holding a clone of the old one.
        self.abort_analysis = Arc::new(AtomicBool::new(false));
        self.eta_secs.store(usize::MAX, Ordering::Relaxed);
        self.is_analyzing.store(true, Ordering::Relaxed);
        let (tx, rx) = std::sync::mpsc::channel();
        self.pre_receiver = Some(rx);
        let sr = self.sample_rate;
        let fft_size    = self.fft_size;
        let pad_factor  = self.pad_factor;
        let window_fn   = self.window_fn.clone();
        let min_freq    = self.min_freq;
        let max_freq    = self.max_freq;
        let interp_mode = self.interp_mode.clone();
        let overlap     = self.overlap;
        let bar_mapping = self.bar_mapping.clone();
        let dsd_rate    = self.dsd_rate;
        let aslt_cfg    = self.aslt_cfg.clone();
        let pre_fps     = self.pre_fps;
        let abort       = Arc::clone(&self.abort_analysis);
        let eta         = Arc::clone(&self.eta_secs);
        let flag = Arc::clone(&self.is_analyzing);
        let progress = Arc::clone(&self.analysis_progress);
        let mapping_dbg = format!("{:?}", self.bar_mapping);
        std::thread::spawn(move || {
            struct ClearOnDrop(Arc<AtomicBool>);
            impl Drop for ClearOnDrop {
                fn drop(&mut self) {
                    self.0.store(false, Ordering::Relaxed);
                    crate::mlog!("[analysis] busy flag cleared");
                }
            }
            let _guard = ClearOnDrop(flag);
            // Logged because a second analysis starting behind the first is
            // invisible from the UI — the progress bar simply appears to restart.
            let t0 = Instant::now();
            crate::mlog!(
                "[analysis] start {mapping_dbg} {} bars @ {pre_fps} fps — {}",
                n_bars,
                path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            );
            let result = preprocess_file(
                &path, &cache, sr, n_bars, &progress,
                fft_size, pad_factor, overlap, &window_fn, min_freq, max_freq, &interp_mode, &bar_mapping,
                dsd_rate, &aslt_cfg, pre_fps, &abort, &eta,
            );
            crate::mlog!(
                "[analysis] {} after {:.1}s",
                match &result {
                    PreMessage::Done { frames, .. } => format!("done, {} frames", frames.len()),
                    PreMessage::Aborted => "aborted".to_string(),
                    PreMessage::Error(e) => format!("error: {e}"),
                },
                t0.elapsed().as_secs_f32(),
            );
            let _ = tx.send(result);
        });
    }

    /// Adopt a freshly loaded or freshly computed set of cached rows.
    ///
    /// The only supported way to install `pre_frames`, because doing so
    /// invalidates the frame cursor: the row index the temporal filter was
    /// continuing from means nothing once the matrix behind it is a different
    /// analysis. There are three places a cache arrives from — a disk load, the
    /// playing-path worker poll, and the paused-path worker poll — and each one
    /// used to assign the field directly.
    pub fn set_pre_frames(&mut self, frames: Vec<Vec<f32>>, frame_rate: f64) {
        self.pre_frames = frames;
        self.pre_frame_rate = frame_rate;
        self.invalidate_pre_cursor();
        self.pre_revision = self.pre_revision.wrapping_add(1);
    }

    /// Drop the cached rows and the cursor into them together.
    pub fn clear_pre_frames(&mut self) {
        self.pre_frames.clear();
        self.invalidate_pre_cursor();
        self.pre_revision = self.pre_revision.wrapping_add(1);
    }

    /// Which cached matrix is installed, as a number that changes when it does.
    pub fn pre_revision(&self) -> u64 {
        self.pre_revision
    }

    /// Ask for the next computed real-time frame to be shown unsmoothed.
    pub fn snap_next_realtime_frame(&mut self) {
        self.snap_next_realtime = true;
    }

    /// Forget where the temporal filter had reached, so the next tick snaps
    /// instead of continuing a cascade that no longer describes anything.
    ///
    /// Called for every discontinuity: track change, seek, backward loop, bar
    /// count change, cache replacement, stop.
    pub fn invalidate_pre_cursor(&mut self) {
        self.last_pre_frame = None;
    }

    /// Push a row directly, for tests that need the ring populated without
    /// driving a whole producer.
    #[cfg(test)]
    pub fn push_waterfall_for_test(&mut self, row: Vec<f32>) {
        self.push_waterfall(row);
    }

    fn push_waterfall(&mut self, row: Vec<f32>) {
        if !self.waterfall_enabled { return; }
        self.waterfall.push(row);
        while self.waterfall.len() > self.waterfall_rows.max(1) {
            self.waterfall.remove(0);
        }
        self.waterfall_dirty = true;
        // Counts rows ever pushed, so the renderer can tell how many are new
        // since it last uploaded. A bare dirty flag cannot: two rows arriving
        // between frames would upload only the newer one and lose the other.
        self.waterfall_seq = self.waterfall_seq.wrapping_add(1);
    }

    /// Resize to a new bar count and clear all derived state.
    /// Drops the receiver so any in-flight thread's send silently fails.
    pub fn set_bar_count(&mut self, n: usize) {
        if n == self.bar_count { return; }
        self.bar_count = n;
        self.magnitudes = vec![0.0; n];
        self.smoothed = vec![0.0; n];
        self.peak_input = vec![0.0; n];
        self.reset_channels();
        self.eq_weights.clear();
        self.waterfall.clear();
        self.waterfall_dirty = false;
        self.clear_pre_frames();
        self.pre_receiver = None;
        // Clearing the flag without cancelling would leave the old run computing
        // a bar count nothing will ever read, while a new one starts beside it.
        self.abort_analysis.store(true, Ordering::Relaxed);
        self.abort_analysis = Arc::new(AtomicBool::new(false));
        self.is_analyzing.store(false, Ordering::Relaxed);
        self.analysis_progress.store(0, Ordering::Relaxed);
        self.eta_secs.store(usize::MAX, Ordering::Relaxed);
    }

    pub fn reset(&mut self) {
        self.reset_inner(true);
    }

    /// Reset the display state but leave a running analysis alone.
    ///
    /// For a track repeating — or being restarted from the top — the analysis
    /// in flight is for exactly the file about to play again, so throwing it
    /// away and starting over means a superlet run can never finish on a track
    /// shorter than itself. On loop it would restart forever.
    pub fn reset_keeping_analysis(&mut self) {
        self.reset_inner(false);
    }

    fn reset_inner(&mut self, cancel_analysis: bool) {
        let n = self.bar_count;
        self.magnitudes = vec![0.0; n];
        self.smoothed = vec![0.0; n];
        self.peak_input = vec![0.0; n];
        self.reset_channels();
        self.snap_next_realtime = true;
        self.eq_weights.clear();
        self.waterfall.clear();
        self.waterfall_dirty = false;
        // Even the keep-the-analysis path is a discontinuity for the display:
        // a repeat restarts at row 0, which is backwards from wherever the
        // cursor had reached.
        self.invalidate_pre_cursor();
        if let Ok(mut b) = self.sample_buf.lock() {
            b.clear();
        }
        if !cancel_analysis {
            // Keep pre_receiver, pre_frames and every flag Arc: dropping the
            // receiver would strand the worker's result and the analysis would
            // run to completion with nothing left to deliver it to.
            return;
        }
        self.clear_pre_frames();
        self.pre_receiver = None;
        // Tell any in-flight analysis to stop *before* the flag Arcs are
        // swapped. Replacing them alone only orphans that thread: it keeps
        // running, keeps a whole CPU busy, and the guard it would have tripped
        // now belongs to nobody — so a new analysis starts alongside it. With a
        // superlet run lasting minutes, two or three of those stack up and every
        // one of them gets slower.
        self.abort_analysis.store(true, Ordering::Relaxed);
        self.abort_analysis = Arc::new(AtomicBool::new(false));
        // Replace the Arcs rather than just storing false/0 into them.
        // Any lingering old-thread ClearOnDrop guard holds a clone of the
        // *previous* Arc; when it fires it writes to that abandoned Arc
        // instead of clobbering the flag for the new analysis we're about
        // to start.
        self.is_analyzing = Arc::new(AtomicBool::new(false));
        self.analysis_progress = Arc::new(AtomicUsize::new(0));
        self.eta_secs = Arc::new(AtomicUsize::new(usize::MAX));
        self.analyzing_path = None;
        self.pending_waveform = None;
        self.pending_analysis = None;
    }

    /// Poll the background pre-process channel and store frames if ready,
    /// without updating magnitudes. Called by SpectrumWindow before the
    /// is_playing guard so results arrive even when paused.
    pub fn try_receive_frames(&mut self) {
        if let Some(ref rx) = self.pre_receiver && let Ok(msg) = rx.try_recv() {
            match msg {
                PreMessage::Done { frames, frame_rate, waveform, analysis } => {
                    // The worker has just written a cache file; keep the whole
                    // directory inside its budget now rather than letting it
                    // grow unbounded between sessions.
                    if self.cache_budget_gb > 0.0 {
                        let budget = (self.cache_budget_gb as f64 * 1e9) as u64;
                        let (n, freed) = evict_cache_to_budget(budget, None);
                        if n > 0 {
                            crate::mlog!("[cache] evicted {n} file(s), freed {:.1} MB",
                                      freed as f64 / 1e6);
                        }
                    }
                    crate::mlog!("[analysis] {} frames received (paused path)", frames.len());
                    self.set_pre_frames(frames, frame_rate);
                    self.pending_waveform = Some(waveform);
                    self.pending_analysis = Some(analysis);
                }
                PreMessage::Error(_) | PreMessage::Aborted => {}
            }
            self.pre_receiver = None;
        }
    }
}

// ---------------------------------------------------------------------------
// Cache helpers
// ---------------------------------------------------------------------------

/// Returns (file_count, total_bytes) for all .spectrumcache files.
fn cache_dir_stats() -> (usize, u64) {
    let dir = home_dir().join(".moosik").join("cache");
    let Ok(entries) = std::fs::read_dir(&dir) else { return (0, 0); };
    let mut count = 0usize;
    let mut bytes = 0u64;
    for e in entries.filter_map(|e| e.ok()) {
        if e.path().extension().map(|x| x == "spectrumcache").unwrap_or(false) {
            count += 1;
            bytes += e.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
    (count, bytes)
}

/// Default ceiling on the whole cache directory, in gigabytes.
///
/// A superlet track at 180 fps and 1024 bars costs 45–80 MB and the encoding
/// cannot be made much smaller — the low byte of every value is quantisation
/// noise, so no lossless coder beats about 12 %, and even truncating to a depth
/// finer than a 4K pixel only reaches 14 %. Bounding the total is therefore the
/// real control, not the packing. 4 GB is roughly 60 superlet tracks, or several
/// hundred FFT-mode ones.
pub const DEFAULT_CACHE_BUDGET_GB: f32 = 4.0;

/// Delete least-recently-used caches until the directory fits `budget_bytes`.
///
/// Returns (files removed, bytes freed). `keep` is spared regardless — it is
/// normally the analysis that just finished, and evicting it immediately would
/// mean a long run wrote a file nobody ever reads.
///
/// LRU also cleans up after cache-key changes for free: caches whose key format
/// no longer exists can never be read, so they are never touched, so they are
/// always first out.
fn evict_cache_to_budget(budget_bytes: u64, keep: Option<&Path>) -> (usize, u64) {
    evict_in_dir(&home_dir().join(".moosik").join("cache"), budget_bytes, keep)
}

fn evict_in_dir(dir: &Path, budget_bytes: u64, keep: Option<&Path>) -> (usize, u64) {
    let Ok(entries) = std::fs::read_dir(dir) else { return (0, 0); };

    let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = Vec::new();
    let mut total = 0u64;
    for e in entries.filter_map(|e| e.ok()) {
        let p = e.path();
        if !p.extension().map(|x| x == "spectrumcache").unwrap_or(false) { continue; }
        let Ok(m) = e.metadata() else { continue };
        // Accessed time is what LRU wants, but it is unreliable — Windows
        // updates it lazily and many Linux mounts disable it outright. Modified
        // time is the honest fallback: for these files it is the time the
        // analysis was written, so it evicts oldest-analysed first.
        let when = m.accessed().or_else(|_| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
        total += m.len();
        files.push((when, m.len(), p));
    }
    if total <= budget_bytes { return (0, 0); }

    files.sort_by_key(|(when, _, _)| *when);
    let mut removed = 0usize;
    let mut freed = 0u64;
    for (_, len, p) in files {
        if total <= budget_bytes { break; }
        if keep.is_some_and(|k| k == p) { continue; }
        if std::fs::remove_file(&p).is_ok() {
            total = total.saturating_sub(len);
            freed += len;
            removed += 1;
        }
    }
    (removed, freed)
}

fn home_dir() -> PathBuf {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

#[allow(clippy::too_many_arguments)]
fn cache_path_for(
    path: &PathBuf,
    n_bars: usize,
    fft_size: usize,
    pad_factor: usize,
    overlap: f32,
    window_fn: &WindowFn,
    min_freq: f32,
    max_freq: f32,
    bar_mapping: &BarMappingMode,
    interp_mode: &InterpolationMode,
    dsd_rate: u32,
    aslt_cfg: &aslt::AsltConfig,
    pre_fps: f32,
) -> PathBuf {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    path.hash(&mut h);
    let hash = h.finish();
    let is_aslt = *bar_mapping == BarMappingMode::Superlet;
    // Superlet ignores every FFT knob, so they are zeroed out of its key rather
    // than baked in — otherwise changing the FFT size while in superlet mode
    // would silently invalidate caches that are still perfectly valid. Nothing
    // can collide with an FFT cache regardless, because `mapping_id` differs.
    let overlap_k  = if is_aslt { 0 } else { (overlap * 1000.0) as u32 };
    let fft_id     = if is_aslt { 0 } else { fft_size };
    let pad_id     = if is_aslt { 0 } else { pad_factor };
    let window_id: u8 = if is_aslt { 0 } else { match window_fn {
        WindowFn::Hann => 0, WindowFn::Hamming => 1,
        WindowFn::Blackman => 2, WindowFn::FlatTop => 3,
    }};
    let mapping_id: u8 = match bar_mapping {
        BarMappingMode::FlatOverlap => 0, BarMappingMode::Gaussian => 1,
        BarMappingMode::Cqt => 2, BarMappingMode::Superlet => 3,
    };
    let interp_id: u8 = if is_aslt { 0 } else { match interp_mode {
        InterpolationMode::None => 0, InterpolationMode::Linear => 1,
        InterpolationMode::CatmullRom => 2, InterpolationMode::Pchip => 3,
        InterpolationMode::Akima => 4, InterpolationMode::Lanczos => 5,
    }};
    let min_hz = min_freq.round() as u32;
    let max_hz = max_freq.round() as u32;
    // DSD analyses depend on the decimation rate, so it joins the key — but
    // only for DSD files, keeping every existing PCM cache filename valid.
    let dsd_part = if crate::dsd::is_dsd_path(path) {
        format!("_d{dsd_rate}")
    } else {
        String::new()
    };
    // Superlet parameters change the frames completely, so they must be in the
    // key. Tenths, because the order ramp is fractional. Frame rate joins too —
    // it is a free parameter here, not derived from the window length.
    let aslt_part = if is_aslt {
        format!(
            "_s{}-{}-{}-{}-{}{}",
            (aslt_cfg.q_ratio * 100.0).round() as u32,
            (aslt_cfg.max_window_s * 100.0).round() as u32,
            aslt_cfg.n_wavelets,
            (aslt_cfg.spread * 100.0).round() as u32,
            pre_fps.round() as u32,
            // Only stamped for the non-default scale, so every cache file
            // written before this existed keeps its name and stays valid.
            match aslt_cfg.scale {
                freq_scale::FreqScale::Log => String::new(),
                freq_scale::FreqScale::Erb => "-erb".to_string(),
                // Tilt in hundredths, so -0.35 and 0.40 get distinct caches.
                freq_scale::FreqScale::Blend(t) => {
                    format!("-w{}", (t * 100.0).round() as i32)
                }
                freq_scale::FreqScale::Lens { tilt, hz, oct, gain } => {
                    format!(
                        "-w{}z{}-{}-{}",
                        (tilt * 100.0).round() as i32,
                        hz.round() as u32,
                        (oct * 10.0).round() as u32,
                        (gain * 10.0).round() as u32,
                    )
                }
            },
        )
    } else {
        String::new()
    };
    home_dir()
        .join(".moosik")
        .join("cache")
        .join(format!(
            "{:016x}_b{}_f{}_w{}_p{}_o{}_n{}_x{}_m{}_i{}{}{}.spectrumcache",
            hash, n_bars, fft_id, window_id, pad_id, overlap_k, min_hz, max_hz,
            mapping_id, interp_id, dsd_part, aslt_part,
        ))
}

// Cache format v2: [magic(4), num_frames(4), num_bars(4), lz4_compressed(u16 LE × n_frames × n_bars)]
// u16 gives 65 536 levels (>30× a 4K screen height). LZ4 adds ~2–4× compression on top.
// Old f32 caches (no magic header) are auto-rejected: their first 4 bytes decode as a frame
// count that won't match CACHE_MAGIC, so they recompute silently.
const CACHE_MAGIC: u32 = 0x4D535032; // "MSP2"

// Cache format v3: same header, but the payload is delta-coded across frequency
// and split into byte planes before LZ4.
//
// Measured on a real 44 645 × 1024 superlet cache: 78.8 MB as v2, 69.1 MB as v3.
// That 12 % is close to the whole prize — the low byte of each u16 is 46 MB of
// uniform quantisation noise, so *no* lossless coder can beat ~77 MB, and even
// truncating to a depth finer than a 4K pixel (12-bit) only reaches 68 MB. The
// file is large because 44 645 frames × 1024 bars is a lot of data, not because
// it is badly packed. Bounding the cache as a whole is the real fix; this is
// just the part that is free.
//
// Neighbouring bars correlate slightly better than consecutive frames, which is
// why the delta runs across frequency — at 180 fps the frames are so similar
// that their difference is dominated by the same noise floor.
const CACHE_MAGIC_V3: u32 = 0x4D535033; // "MSP3"

fn load_cache(cache_path: &PathBuf, n_bars: usize) -> Option<Vec<Vec<f32>>> {
    use std::io::Read;
    let mut file = std::fs::File::open(cache_path).ok()?;
    let mut raw = Vec::new();
    file.read_to_end(&mut raw).ok()?;
    if raw.len() < 12 { return None; }

    let magic      = u32::from_le_bytes(raw[0..4].try_into().ok()?);
    let num_frames = u32::from_le_bytes(raw[4..8].try_into().ok()?) as usize;
    let num_bars   = u32::from_le_bytes(raw[8..12].try_into().ok()?) as usize;

    let v3 = magic == CACHE_MAGIC_V3;
    if (!v3 && magic != CACHE_MAGIC) || num_bars != n_bars
        || num_bars > MAX_BAR_COUNT || num_frames > 500_000
    {
        return None;
    }

    let decompressed = lz4_flex::decompress_size_prepended(&raw[12..]).ok()?;

    // Each value is a u16 → 2 bytes per bar per frame.
    let count = num_frames.checked_mul(num_bars)?;
    let expected_bytes = count.checked_mul(2)?;
    if decompressed.len() < expected_bytes { return None; }

    if !v3 {
        // v2: plain u16 LE, interleaved.
        return Some((0..num_frames)
            .map(|f| {
                let base = f * num_bars * 2;
                (0..num_bars).map(|b| {
                    let off = base + b * 2;
                    let v = u16::from_le_bytes([decompressed[off], decompressed[off + 1]]);
                    v as f32 / 65535.0
                }).collect()
            })
            .collect());
    }

    // v3: high-byte plane, then low-byte plane, each holding a zigzag delta
    // taken across frequency within a frame.
    let (hi, lo) = decompressed.split_at(count);
    let mut frames = Vec::with_capacity(num_frames);
    for f in 0..num_frames {
        let base = f * num_bars;
        let mut row = Vec::with_capacity(num_bars);
        let mut prev: i32 = 0;
        for b in 0..num_bars {
            let z = u16::from_le_bytes([lo[base + b], hi[base + b]]) as u32;
            let d = ((z >> 1) as i32) ^ -((z & 1) as i32);
            prev = (prev + d) & 0xFFFF;
            row.push(prev as f32 / 65535.0);
        }
        frames.push(row);
    }
    Some(frames)
}

fn save_cache(cache_path: &PathBuf, frames: &[Vec<f32>]) {
    use std::io::Write;
    if let Some(parent) = cache_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let n_bars = frames.first().map(|f| f.len()).unwrap_or(0);

    // Zigzag delta across frequency, then split the two bytes of every value
    // into separate planes. Interleaved, the noisy low byte sits between every
    // pair of compressible high bytes and stops LZ4 finding any run at all.
    let count = frames.len() * n_bars;
    let mut hi: Vec<u8> = Vec::with_capacity(count);
    let mut lo: Vec<u8> = Vec::with_capacity(count);
    for frame in frames {
        let mut prev: i32 = 0;
        for &v in frame {
            let q = (v.clamp(0.0, 1.0) * 65535.0).round() as i32;
            // Wrap the difference into i16 before zigzagging. The raw difference
            // spans ±65535 and needs 17 bits; wrapped, it still reconstructs
            // exactly because the values themselves are 16-bit, and it keeps the
            // common small steps small — which is the entire point of the delta.
            let d = (q - prev) as i16 as i32;
            prev = q;
            let z = ((d << 1) ^ (d >> 31)) as u32 as u16;
            let [b0, b1] = z.to_le_bytes();
            lo.push(b0);
            hi.push(b1);
        }
    }
    let mut payload = hi;
    payload.append(&mut lo);
    let compressed = lz4_flex::compress_prepend_size(&payload);

    let mut header = Vec::with_capacity(12 + compressed.len());
    header.extend_from_slice(&CACHE_MAGIC_V3.to_le_bytes());
    header.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    header.extend_from_slice(&(n_bars as u32).to_le_bytes());
    header.extend_from_slice(&compressed);

    if let Ok(mut f) = std::fs::File::create(cache_path) && f.write_all(&header).is_err() {
        let _ = std::fs::remove_file(cache_path);
    }
}

// ---------------------------------------------------------------------------
// Background pre-processing (uses rodio Decoder directly)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn preprocess_file(
    path: &PathBuf,
    cache_path: &PathBuf,
    _sample_rate_hint: u32,
    n_bars: usize,
    progress: &Arc<AtomicUsize>,
    fft_size: usize,
    pad_factor: usize,
    overlap: f32,
    window_fn: &WindowFn,
    min_freq: f32,
    max_freq: f32,
    interp_mode: &InterpolationMode,
    bar_mapping: &BarMappingMode,
    dsd_rate: u32,
    aslt_cfg: &aslt::AsltConfig,
    pre_fps: f32,
    abort: &Arc<AtomicBool>,
    eta_secs: &Arc<AtomicUsize>,
) -> PreMessage {
    // ── Phase 1: decode all mono samples sequentially (0–49 %) ──────────────
    // DSD has no PCM samples to decode — it's low-pass-filtered and decimated
    // to the analysis rate instead (the audible band survives; the ultrasonic
    // modulator noise the analyzer shouldn't show is what the filter removes).
    let (all_mono, sample_rate) = if crate::dsd::is_dsd_path(path) {
        let mut src = match crate::dsd::decimate::open_pcm_source(path, dsd_rate) {
            Ok(s) => s,
            Err(e) => return PreMessage::Error(e),
        };
        let sr = src.out_rate();
        let ch = src.channels().max(1);
        let total_hint = (src.total_out_samples() / ch as u64) as usize;
        let mut mono: Vec<f32> = Vec::with_capacity(total_hint.max(1));
        loop {
            let mut sum = 0.0f32;
            let mut got = 0usize;
            for _ in 0..ch {
                match src.next() {
                    Some(s) => { sum += s; got += 1; }
                    None    => break,
                }
            }
            if got == 0 { break; }
            mono.push(sum / got as f32);
            if total_hint > 0 {
                progress.store(
                    (mono.len() * PROG_DECODE_END / total_hint).min(PROG_DECODE_END),
                    Ordering::Relaxed,
                );
            }
        }
        (mono, sr)
    } else {
        use rodio::Decoder;
        use std::io::BufReader;

        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) => return PreMessage::Error(e.to_string()),
        };
        let decoder = match Decoder::new(BufReader::new(file)) {
            Ok(d) => d,
            Err(e) => return PreMessage::Error(e.to_string()),
        };

        let sr = decoder.sample_rate();
        let ch = (decoder.channels() as usize).max(1);

        // Estimate total mono samples for progress reporting (best-effort).
        let total_hint = decoder.total_duration()
            .map(|d| (d.as_secs_f64() * sr as f64) as usize)
            .unwrap_or(0);

        let mut mono: Vec<f32> = Vec::with_capacity(total_hint.max(1));
        let mut raw_iter = decoder;
        loop {
            let mut sum = 0.0f32;
            let mut got = 0usize;
            for _ in 0..ch {
                match raw_iter.next() {
                    Some(s) => { sum += s as f32 / 32_768.0; got += 1; }
                    None    => break,
                }
            }
            if got == 0 { break; }
            mono.push(sum / got as f32);
            if total_hint > 0 {
                progress.store(
                    (mono.len() * PROG_DECODE_END / total_hint).min(PROG_DECODE_END),
                    Ordering::Relaxed,
                );
            }
        }
        (mono, sr)
    };

    // The superlet has no analysis window to derive a frame rate from, so its
    // hop comes straight from the target frame rate instead of from
    // window × overlap.
    let is_aslt = *bar_mapping == BarMappingMode::Superlet;
    let hop = if is_aslt {
        aslt::hop_for_fps(sample_rate, pre_fps)
    } else {
        ((fft_size as f32 * (1.0 - overlap)).round() as usize).max(1)
    };
    let padded_size = fft_size * pad_factor;
    let window = make_window(fft_size, window_fn);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(padded_size);
    let half = padded_size / 2;
    let scale = fft_size as f32; // normalize by window length, not padded length
    let log_min = min_freq.log10();
    let log_max = (sample_rate as f32 / 2.0).min(max_freq).log10();

    // Superlet needs no particular window length, but the loudness/chroma pass
    // below still runs a 4096-point FFT, so that is the real floor.
    let min_len = if is_aslt { CHROMA_FFT } else { fft_size };
    if all_mono.len() < min_len {
        return PreMessage::Error("audio too short for analysis".into());
    }
    progress.store(PROG_DECODE_END + 1, Ordering::Relaxed);

    // ── Waveform + analysis pass (single-threaded over already-decoded data) ──
    let waveform_n_cols = 1000usize;
    let mut kw = KWeightFilter::new(sample_rate);

    const CHROMA_FFT: usize = 4096;
    const CHROMA_HOP: usize = 2048;
    let chroma_half = CHROMA_FFT / 2;
    let mut chroma_planner = FftPlanner::<f32>::new();
    let chroma_fft_plan = chroma_planner.plan_fft_forward(CHROMA_FFT);
    let chroma_win = make_window(CHROMA_FFT, &WindowFn::Hann);
    let mut fft_ring: Vec<f32> = Vec::with_capacity(CHROMA_FFT + CHROMA_HOP);
    let mut chroma_total = [0.0f32; 12];
    let mut prev_fft_norms = vec![0.0f32; chroma_half];
    let mut flux_series: Vec<f32> = Vec::new();
    let flux_rate = sample_rate as f32 / CHROMA_HOP as f32;

    let lhist_block = sample_rate as usize;
    let mut lhist_sq = 0.0f64;
    let mut lhist_n  = 0usize;
    let mut loudness_history: Vec<f32> = Vec::new();

    const CHUNK: usize = 1024;
    let mut chunks: Vec<f32> = Vec::new();
    let mut rms_sq = 0.0f32;
    let mut rms_count = 0usize;

    let block_size = (3 * sample_rate) as usize;
    let mut dr_blocks: Vec<(f32, f32)> = Vec::new();
    let mut blk_peak = 0.0f32;
    let mut blk_sq   = 0.0f32;
    let mut blk_n    = 0usize;

    let gate_size = (sample_rate as f64 * 0.4) as usize;
    let mut lufs_blocks: Vec<f32> = Vec::new();
    let mut gate_sq = 0.0f64;
    let mut gate_n  = 0usize;

    let mut clip_count = 0u32;
    let mut clip_positions: Vec<f32> = Vec::new();
    const CLIP_THRESH: f32 = 0.9999;
    let total_samples = all_mono.len();

    // One update per 1 % of the pass; per-sample would be millions of atomic
    // stores for a bar that only moves ten steps.
    let loudness_step = (total_samples / 100).max(1);
    for (idx, &s) in all_mono.iter().enumerate() {
        if idx % loudness_step == 0 && total_samples > 0 {
            let span = PROG_LOUDNESS_END - PROG_DECODE_END;
            progress.store(
                PROG_DECODE_END + (idx * span / total_samples).min(span),
                Ordering::Relaxed,
            );
        }
        rms_sq += s * s;
        rms_count += 1;
        if rms_count >= CHUNK {
            chunks.push((rms_sq / rms_count as f32).sqrt());
            rms_sq = 0.0; rms_count = 0;
        }
        if s.abs() >= CLIP_THRESH {
            clip_count += 1;
            if clip_positions.len() < 500 {
                clip_positions.push(if total_samples > 0 { idx as f32 / total_samples as f32 } else { 0.0 });
            }
        }
        blk_peak = blk_peak.max(s.abs());
        blk_sq  += s * s;
        blk_n   += 1;
        if blk_n >= block_size {
            dr_blocks.push((blk_peak, blk_sq / blk_n as f32));
            blk_peak = 0.0; blk_sq = 0.0; blk_n = 0;
        }
        let kw_s = kw.process(s);
        gate_sq += (kw_s * kw_s) as f64;
        gate_n  += 1;
        if gate_n >= gate_size {
            lufs_blocks.push((gate_sq / gate_n as f64) as f32);
            gate_sq = 0.0; gate_n = 0;
        }
        lhist_sq += (kw_s * kw_s) as f64;
        lhist_n  += 1;
        if lhist_n >= lhist_block {
            let lufs_s = if lhist_sq / lhist_n as f64 > 1e-10 {
                (-0.691 + 10.0 * (lhist_sq / lhist_n as f64).log10()) as f32
            } else { -70.0 };
            loudness_history.push(lufs_s);
            lhist_sq = 0.0; lhist_n = 0;
        }
        fft_ring.push(s);
        if fft_ring.len() >= CHROMA_FFT {
            let mut buf: Vec<Complex<f32>> = fft_ring[..CHROMA_FFT].iter()
                .zip(chroma_win.iter())
                .map(|(x, w)| Complex { re: x * w, im: 0.0 })
                .collect();
            chroma_fft_plan.process(&mut buf);
            let norms: Vec<f32> = buf[..chroma_half].iter().map(|c| c.norm()).collect();
            let flux: f32 = norms.iter().zip(&prev_fft_norms).map(|(&a, &b)| (a - b).max(0.0)).sum();
            flux_series.push(flux);
            prev_fft_norms.copy_from_slice(&norms);
            let fc = compute_chroma(&norms, sample_rate, CHROMA_FFT);
            for (a, &b) in chroma_total.iter_mut().zip(&fc) { *a += b; }
            fft_ring.drain(..CHROMA_HOP);
        }
    }
    // Flush partials
    if rms_count > 0 { chunks.push((rms_sq / rms_count as f32).sqrt()); }
    if blk_n     > 0 { dr_blocks.push((blk_peak, blk_sq / blk_n as f32)); }
    if gate_n    > 0 { lufs_blocks.push((gate_sq / gate_n as f64) as f32); }
    if lhist_n   > 0 {
        let lufs_s = if lhist_sq / lhist_n as f64 > 1e-10 {
            (-0.691 + 10.0 * (lhist_sq / lhist_n as f64).log10()) as f32
        } else { -70.0 };
        loudness_history.push(lufs_s);
    }

    let waveform: Vec<f32> = if chunks.is_empty() {
        vec![0.0; waveform_n_cols]
    } else {
        let peak = chunks.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
        (0..waveform_n_cols).map(|i| {
            let idx = (i * chunks.len() / waveform_n_cols).min(chunks.len() - 1);
            (chunks[idx] / peak).clamp(0.0, 1.0)
        }).collect()
    };

    let peak_dbfs = dr_blocks.iter().map(|&(p, _)| p).fold(0.0f32, f32::max);
    let peak_db   = if peak_dbfs > 0.0 { 20.0 * peak_dbfs.log10() } else { -144.0 };
    let dr_score = if dr_blocks.len() >= 2 {
        dr_blocks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_n = ((dr_blocks.len() as f32 * 0.2).ceil() as usize).max(1);
        let loud_mean_sq = dr_blocks[..top_n].iter().map(|&(_, sq)| sq).sum::<f32>() / top_n as f32;
        let loud_rms_db = if loud_mean_sq > 0.0 { 20.0 * loud_mean_sq.sqrt().log10() } else { -144.0 };
        ((peak_db - loud_rms_db).round() as i32).clamp(0, 20) as u32
    } else { 0 };

    let abs_gate_sq = 10_f32.powf((-70.0 + 0.691) / 10.0);
    let gated: Vec<f32> = lufs_blocks.into_iter().filter(|&sq| sq > abs_gate_sq).collect();
    let integrated_lufs = if gated.is_empty() {
        f32::NEG_INFINITY
    } else {
        let mean_sq = gated.iter().sum::<f32>() / gated.len() as f32;
        -0.691 + 10.0 * (mean_sq as f64).log10() as f32
    };

    let key_name = detect_key(&chroma_total);
    let bpm = detect_bpm(&flux_series, flux_rate);

    let analysis = TrackAnalysis {
        integrated_lufs, dr_score, peak_dbfs: peak_db,
        clip_count, clip_positions, bpm, key_name,
        loudness_history,
    };

    // ── Phase 2a: superlet (50–99 %) ─────────────────────────────────────────
    // Wholly replaces the FFT pipeline — no bins, no window, no bar mapping.
    // The frame layout it produces is identical, so the cache format, the
    // renderer and the waterfall are all untouched.
    if is_aslt {
        use std::sync::atomic::AtomicU64;

        // Same bar grid as the FFT path, Nyquist clamp included, so switching
        // modes does not shift the bars sideways.
        let a_max = (sample_rate as f32 / 2.0).min(max_freq);
        let per_frame = aslt::bar_taps_per_frame(sample_rate, n_bars, min_freq, a_max, aslt_cfg);
        let n_frames = aslt::frame_count(all_mono.len(), hop) as f64;
        let total_taps = (per_frame.iter().sum::<f64>() * n_frames).max(1.0);

        let done_taps = AtomicU64::new(0);
        let started = Instant::now();
        // (taps at last sample, elapsed at last sample, smoothed taps/sec).
        // A mutex rather than atomics because the three move together and this
        // is touched once per finished bar, not per frame.
        let rate_window = std::sync::Mutex::new((0.0f64, 0.0f64, 0.0f64));
        eta_secs.store((total_taps / aslt::TAPS_PER_SEC_HINT) as usize, Ordering::Relaxed);

        // Held to the user's core budget. `install` makes that pool current, so
        // every nested `par_iter` inside the transform inherits the limit.
        let raw = gpu_calib::install(|| aslt::analyze_with_progress(
            &all_mono, sample_rate, n_bars, min_freq, a_max, hop, aslt_cfg,
            &|| !abort.load(Ordering::Relaxed),
            &|bar| {
                let cost = per_frame[bar] * n_frames;
                let done = done_taps.fetch_add(cost as u64, Ordering::Relaxed) as f64 + cost;
                store_progress_max(
                    progress,
                    50 + ((done / total_taps).clamp(0.0, 1.0) * 49.0) as usize,
                );
                // Rate over the last second or so, not since the run started.
                //
                // A cumulative average assumes the work ahead costs what the
                // work behind did, and here it does not: the frequency-domain
                // bars finish first and the short-kernel bars, which are
                // computed frame by frame, come last. A tap on that route costs
                // more wall clock than a tap on the transform route, so the
                // average was always flattering and the last stretch always
                // overran its own estimate. Measuring recent throughput instead
                // lets the estimate notice the slowdown while it is happening,
                // which is the difference between an estimate and a guess.
                let elapsed = started.elapsed().as_secs_f64();
                let recent = {
                    let mut g = rate_window.lock().unwrap_or_else(|e| e.into_inner());
                    let (last_done, last_at, ema) = *g;
                    if elapsed - last_at >= 1.0 && done > last_done {
                        let inst = (done - last_done) / (elapsed - last_at);
                        // Smoothed, because bars land in bursts as the pool
                        // drains and a raw sample swings wildly.
                        let next = if ema <= 0.0 { inst } else { ema * 0.6 + inst * 0.4 };
                        *g = (done, elapsed, next);
                        next
                    } else {
                        ema
                    }
                };
                let rate = if recent > 0.0 {
                    recent
                } else if done > total_taps * 0.02 && elapsed > 0.5 {
                    done / elapsed
                } else {
                    aslt::TAPS_PER_SEC_HINT
                };
                eta_secs.store(((total_taps - done) / rate).max(0.0) as usize, Ordering::Relaxed);
            },
        ));

        if abort.load(Ordering::Relaxed) { return PreMessage::Aborted; }
        if raw.is_empty() {
            return PreMessage::Error("superlet analysis produced no frames".into());
        }

        let frames: Vec<Vec<f32>> = raw
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(bar, &mag)| {
                        let _ = bar;
                        let db = (20.0 * (mag * ASLT_DB_REF).log10()).max(-80.0);
                        ((db + 80.0) / 80.0).clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect();

        progress.store(100, Ordering::Relaxed);
        eta_secs.store(0, Ordering::Relaxed);
        let frame_rate = sample_rate as f64 / hop as f64;
        save_cache(cache_path, &frames);
        return PreMessage::Done { frames, frame_rate, waveform, analysis };
    }

    // ── Phase 2: process frames in parallel with rayon (50–99 %) ─────────────
    let num_frames = (all_mono.len() - fft_size) / hop + 1;
    let done = AtomicUsize::new(0);
    let nf   = num_frames.max(1);

    // Pre-compute CQT Q factor (used only when bar_mapping == Cqt)
    let n_octaves_pp = (log_max - log_min) / 2_f32.log10();
    let bins_per_oct_pp = n_bars as f32 / n_octaves_pp.max(0.1);
    let cqt_q_pp = 1.0 / (2.0_f32.powf(1.0 / bins_per_oct_pp) - 1.0);

    use rayon::prelude::*;
    let frames: Vec<Vec<f32>> = (0..num_frames)
        .into_par_iter()
        .map(|fi| {
            let start = fi * hop;
            let slice = &all_mono[start..start + fft_size];

            let mut buf: Vec<Complex<f32>> = slice.iter()
                .zip(window.iter())
                .map(|(s, w)| Complex { re: s * w, im: 0.0 })
                .chain(std::iter::repeat_n(Complex { re: 0.0, im: 0.0 }, padded_size - fft_size))
                .collect();
            fft.process(&mut buf);
            let norms: Vec<f32> = buf[..half].iter().map(|c| c.norm()).collect();

            let bars: Vec<f32> = (0..n_bars)
                .map(|bar| {
                    let fscale = aslt_cfg.scale;
                    let mag = if *bar_mapping == BarMappingMode::Cqt {
                        let tc  = (bar as f32 + 0.5) / n_bars as f32;
                        let f_c = fscale.freq_at(tc, min_freq, max_freq);
                        let bc  = (f_c * padded_size as f32 / sample_rate as f32).clamp(1.0, half as f32 - 1.0);
                        cqt_kernel(&norms, bc, cqt_q_pp, half)
                    } else {
                        let t0 = bar as f32 / n_bars as f32;
                        let t1 = (bar + 1) as f32 / n_bars as f32;
                        let freq_lo = fscale.freq_at(t0, min_freq, max_freq);
                        let freq_hi = fscale.freq_at(t1, min_freq, max_freq);
                        let fbin_lo = (freq_lo * padded_size as f32 / sample_rate as f32).max(1.0);
                        let fbin_hi = (freq_hi * padded_size as f32 / sample_rate as f32)
                            .max(fbin_lo + 0.001).min(half as f32 - 0.001);
                        if fbin_hi - fbin_lo <= 1.0 {
                            let center = (fbin_lo + fbin_hi) * 0.5;
                            interp_sub_bin(&norms, center, interp_mode)
                        } else {
                            let b_start = fbin_lo.floor() as usize;
                            let b_end   = (fbin_hi.ceil() as usize).min(half - 1);
                            let bc      = (fbin_lo + fbin_hi) * 0.5;
                            let sigma   = ((fbin_hi - fbin_lo) * 0.5).max(0.5);
                            let mut wsum = 0.0_f32;
                            let mut weight = 0.0_f32;
                            for (b_idx, &norm_b) in norms[b_start..=b_end].iter().enumerate() {
                                let b = b_start + b_idx;
                                let w = match bar_mapping {
                                    BarMappingMode::FlatOverlap => {
                                        (fbin_hi.min(b as f32 + 1.0) - fbin_lo.max(b as f32)).max(0.0)
                                    }
                                    BarMappingMode::Gaussian
                                    | BarMappingMode::Cqt
                                    | BarMappingMode::Superlet => {
                                        let center_b = b as f32 + 0.5;
                                        (-(center_b - bc).powi(2) / (2.0 * sigma * sigma)).exp()
                                    }
                                };
                                wsum += norm_b * w; weight += w;
                            }
                            if weight > 0.0 { wsum / weight } else { 0.0 }
                        }
                    };
                    let db = 20.0 * (mag / scale).log10().max(-80.0);
                    ((db + 80.0) / 80.0).clamp(0.0, 1.0)
                })
                .collect();

            let c = done.fetch_add(1, Ordering::Relaxed) + 1;
            progress.store(50 + (c * 49 / nf).min(49), Ordering::Relaxed);
            bars
        })
        .collect();

    // The FFT path is fast enough that Abort rarely gets a chance to fire, but
    // it must still honour it rather than write a cache the user cancelled.
    if abort.load(Ordering::Relaxed) { return PreMessage::Aborted; }

    progress.store(100, Ordering::Relaxed);

    let frame_rate = sample_rate as f64 / hop as f64;
    save_cache(cache_path, &frames);
    PreMessage::Done { frames, frame_rate, waveform, analysis }
}

// ---------------------------------------------------------------------------
// Spectral ceiling helpers
// ---------------------------------------------------------------------------

/// Richer result from spectral analysis — ceiling frequency plus rolloff shape.
#[derive(Clone, Debug)]
pub struct SpectralCeiling {
    /// Highest frequency with meaningful energy (Hz).
    pub hz: f32,
    /// Width of the rolloff region in octaves. Small (< 0.35) = brick-wall;
    /// large (> 1.0) = gradual/natural. f32::INFINITY = no clear cutoff found.
    pub rolloff_octaves: f32,
    /// Standard sample rate whose Nyquist matches `hz` within 8 %, if any.
    pub matched_standard_sr: Option<u32>,
}

/// Measure how many octaves it takes for the peak spectrum to drop from 40 % of
/// the global peak down to the noise floor. Small values indicate a brick-wall
/// filter (characteristic of digital upsampling); large values indicate a
/// natural, gradual rolloff (e.g. synth music, acoustic instruments).
fn rolloff_width_octaves(peak: &[f32], min_freq: f32, max_freq: f32) -> f32 {
    let n = peak.len();
    if n == 0 { return f32::INFINITY; }
    let global_peak = peak.iter().cloned().fold(0.0f32, f32::max);
    if global_peak < 0.1 { return f32::INFINITY; }

    let log_min = min_freq.log10();
    let log_max = max_freq.log10();
    let bar_to_log_freq = |i: usize| -> f32 {
        log_min + (i as f32 + 0.5) / n as f32 * (log_max - log_min)
    };

    let high_threshold = global_peak * 0.4;
    let low_threshold  = (global_peak * 0.05).max(0.04);

    // Highest bar still above the "content present" threshold
    let high_bar = match (0..n).rev().find(|&i| peak[i] > high_threshold) {
        Some(h) => h,
        None => return f32::INFINITY,
    };
    // Scan downward from there to find where energy drops to noise floor
    let low_bar = match (0..=high_bar).rev().find(|&i| peak[i] < low_threshold) {
        Some(lo) => lo,
        None => return f32::INFINITY, // no noise floor found — content fills the range
    };

    let log_hi = bar_to_log_freq(high_bar);
    let log_lo = bar_to_log_freq(low_bar + 1);
    if log_hi <= log_lo { return 0.0; }
    // Convert log10 difference to octaves
    (log_hi - log_lo) / 2_f32.log10()
}

/// Return the standard sample rate (22 050, 32 000, 44 100, 48 000 Hz) whose
/// Nyquist is within 8 % of `ceiling_hz`, or None.
fn matches_standard_nyquist(ceiling_hz: f32) -> Option<u32> {
    const RATES: &[u32] = &[22_050, 32_000, 44_100, 48_000];
    for &sr in RATES {
        let nyquist = sr as f32 / 2.0;
        if (ceiling_hz - nyquist).abs() / nyquist < 0.08 {
            return Some(sr);
        }
    }
    None
}

/// Analyse pre-processed FFT frames and return spectral ceiling + rolloff shape.
fn compute_spectral_ceiling(
    frames: &[Vec<f32>], n_bars: usize, min_freq: f32, max_freq: f32,
    scale: freq_scale::FreqScale,
) -> Option<SpectralCeiling> {
    if frames.is_empty() || n_bars == 0 { return None; }
    let noise = 0.15_f32;
    let mut peak = vec![0.0f32; n_bars];
    for frame in frames {
        for (i, &v) in frame.iter().enumerate().take(n_bars) {
            if v > peak[i] { peak[i] = v; }
        }
    }
    let highest = peak.iter().enumerate().rev()
        .find(|&(_, &v)| v > noise).map(|(i, _)| i)?;
    let hz = scale.bar_center(highest, n_bars, min_freq, max_freq);
    let rolloff_octaves      = rolloff_width_octaves(&peak, min_freq, max_freq);
    let matched_standard_sr  = matches_standard_nyquist(hz);
    Some(SpectralCeiling { hz, rolloff_octaves, matched_standard_sr })
}

// ---------------------------------------------------------------------------
// Spectrogram (pixel-accurate rolling time-frequency display)
//
// Currently real-time only: each frame stores 4096 raw norms vs. 64 bars for
// the spectrum cache, making pre-process support a 64× storage increase plus
// a cache format redesign. The author is open to tackling this if a compact
// representation (e.g. log-quantised, delta-coded) keeps the cache size
// reasonable. For now the view is hidden in pre-process mode.
// ---------------------------------------------------------------------------

const SPEC_W: usize = 600; // time columns
const SPEC_H: usize = 256; // frequency rows

struct Spectrogram {
    pixels: Vec<Color32>, // SPEC_H rows × SPEC_W cols, row-major
    col_head: usize,      // ring-buffer write pointer
    pub dirty: bool,
}

impl Spectrogram {
    fn new() -> Self {
        Self {
            pixels: vec![Color32::BLACK; SPEC_W * SPEC_H],
            col_head: 0,
            dirty: false,
        }
    }

    /// Add one time-column from raw half-spectrum norms.
    fn push_frame(&mut self, norms: &[f32], sr: u32, fft_size: usize, min_freq: f32, max_freq: f32, pal: &Palette) {
        let half  = norms.len();
        let scale = fft_size as f32;
        let log_min = min_freq.log10();
        let log_max = (sr as f32 / 2.0).min(max_freq).log10();
        for row in 0..SPEC_H {
            // row 0 = top = high freq; row SPEC_H-1 = bottom = low freq
            let t = (SPEC_H - 1 - row) as f32 / (SPEC_H - 1) as f32;
            let freq = 10_f32.powf(log_min + t * (log_max - log_min));
            let bin  = ((freq * fft_size as f32 / sr as f32) as usize).clamp(1, half - 1);
            let db   = 20.0 * (norms[bin] / scale).log10().max(-80.0);
            let v    = ((db + 80.0) / 80.0).clamp(0.0, 1.0);
            self.pixels[row * SPEC_W + self.col_head] = pal.heat(v);
        }
        self.col_head = (self.col_head + 1) % SPEC_W;
        self.dirty = true;
    }

    /// Reorder ring buffer into a linear image (oldest column → left).
    fn to_color_image(&self) -> egui::ColorImage {
        let mut ordered = vec![Color32::BLACK; SPEC_W * SPEC_H];
        for row in 0..SPEC_H {
            for ci in 0..SPEC_W {
                let src = (self.col_head + ci) % SPEC_W;
                ordered[row * SPEC_W + ci] = self.pixels[row * SPEC_W + src];
            }
        }
        egui::ColorImage { size: [SPEC_W, SPEC_H], pixels: ordered }
    }

    fn clear(&mut self) {
        self.pixels.fill(Color32::BLACK);
        self.col_head = 0;
        self.dirty = true;
    }
}

// ---------------------------------------------------------------------------
// ISO 1/3-octave RTA
// ---------------------------------------------------------------------------

/// ISO 1/3-octave centre frequencies (Hz), 20 Hz – 20 kHz.
const ISO_THIRD_OCTAVE: &[f32] = &[
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
    200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0,
    1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0,
    10_000.0, 12_500.0, 16_000.0, 20_000.0,
];

/// Compute per-band peak magnitude (0–1) from raw half-spectrum norms.
fn octave_band_magnitudes(norms: &[f32], sr: u32, fft_size: usize) -> Vec<(f32, f32)> {
    let half    = norms.len();
    let nyquist = sr as f32 / 2.0;
    let scale   = fft_size as f32;
    ISO_THIRD_OCTAVE.iter()
        .filter(|&&fc| fc < nyquist * 0.95)
        .map(|&fc| {
            let lo = (fc * 2_f32.powf(-1.0 / 6.0)).max(1.0);
            let hi = (fc * 2_f32.powf( 1.0 / 6.0)).min(nyquist);
            let bin_lo = ((lo * fft_size as f32 / sr as f32) as usize).clamp(1, half - 1);
            let bin_hi = ((hi * fft_size as f32 / sr as f32) as usize).clamp(bin_lo, half - 1);
            let peak = norms[bin_lo..=bin_hi].iter().cloned().fold(0.0_f32, f32::max);
            let db   = 20.0 * (peak / scale).log10().max(-80.0);
            let v    = ((db + 80.0) / 80.0).clamp(0.0, 1.0);
            (fc, v)
        })
        .collect()
}

fn draw_octave_bands(
    painter: &egui::Painter, bands: &[(f32, f32)],
    plot_rect: Rect, sr: u32, min_freq: f32, max_freq: f32, pal: &Palette,
) {
    if bands.is_empty() { return; }
    let nyquist  = sr as f32 / 2.0;
    let log_min  = min_freq.log10();
    let log_max  = nyquist.min(max_freq).log10();
    let log_span = (log_max - log_min).max(1e-6);

    for &(fc, mag) in bands {
        if fc <= 0.0 { continue; }
        let lo = fc * 2_f32.powf(-1.0 / 6.0);
        let hi = fc * 2_f32.powf( 1.0 / 6.0);
        let t_lo = ((lo.log10() - log_min) / log_span).clamp(0.0, 1.0);
        let t_hi = ((hi.log10() - log_min) / log_span).clamp(0.0, 1.0);
        let t_c  = ((fc.log10()  - log_min) / log_span).clamp(0.0, 1.0);
        let x_lo = plot_rect.left() + t_lo * plot_rect.width();
        let x_hi = plot_rect.left() + t_hi * plot_rect.width();
        let x_c  = plot_rect.left() + t_c  * plot_rect.width();

        let bar_h = mag * plot_rect.height();
        painter.rect_filled(
            Rect::from_min_max(
                Pos2::new(x_lo + 1.0, plot_rect.bottom() - bar_h),
                Pos2::new((x_hi - 1.0).max(x_lo + 2.0), plot_rect.bottom()),
            ),
            0.0, pal.bar(mag),
        );

        // Frequency label below the plot area
        let label = if fc >= 1000.0 { format!("{:.0}k", fc / 1000.0) }
                    else             { format!("{fc:.0}") };
        painter.text(
            Pos2::new(x_c, plot_rect.bottom() + 5.0),
            egui::Align2::CENTER_TOP,
            label,
            egui::FontId::monospace(7.0),
            Color32::from_gray(90),
        );
    }
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

/// Constant-Q Hann-windowed kernel.
/// `bc` = centre bin (fractional), `q` = Q factor = f/Δf.
/// When the bandwidth is sub-bin (bc/q < 1), degrades gracefully to nearest-bin.
fn cqt_kernel(norms: &[f32], bc: f32, q: f32, half: usize) -> f32 {
    let hw = (bc / q).max(0.5); // half-lobe width in bins
    let b_lo = ((bc - hw).floor() as isize).max(0) as usize;
    let b_hi = ((bc + hw).ceil()  as usize).min(half - 1);
    let mut wsum   = 0.0f32;
    let mut weight = 0.0f32;
    for (b_idx, &norm_b) in norms[b_lo..=b_hi].iter().enumerate() {
        let b = b_lo + b_idx;
        let x = (b as f32 + 0.5 - bc) / hw; // normalised to [−1, 1]
        if x.abs() < 1.0 {
            let w = 0.5 * (1.0 + (PI * x).cos()); // Hann
            wsum   += norm_b * w;
            weight += w;
        }
    }
    if weight > 0.0 { (wsum / weight).max(0.0) } else { norms[bc.round() as usize % half] }
}

fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t; let t3 = t2 * t;
    0.5 * ((2.0*p1) + (-p0+p2)*t + (2.0*p0-5.0*p1+4.0*p2-p3)*t2 + (-p0+3.0*p1-3.0*p2+p3)*t3)
}

// ---------------------------------------------------------------------------
// Interpolation helpers
// ---------------------------------------------------------------------------

#[inline]
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-6 { 1.0 } else { (PI * x).sin() / (PI * x) }
}

#[inline]
fn lanczos_kernel(x: f32, a: f32) -> f32 {
    if x.abs() >= a { 0.0 } else { sinc(x) * sinc(x / a) }
}

/// Fritsch-Carlson PCHIP tangent for uniform spacing.
/// Returns 0 when neighbouring slopes change sign (prevents overshoot).
#[inline]
fn pchip_slope(d0: f32, d1: f32) -> f32 {
    if d0 * d1 <= 0.0 { return 0.0; }
    // Harmonic mean — inherently monotone-preserving
    let s = 2.0 * d0 * d1 / (d0 + d1);
    // Fritsch-Carlson limiter: clamp to 3× the smaller delta
    if s.abs() > 3.0 * d0.abs().min(d1.abs()) {
        3.0 * d0.abs().min(d1.abs()) * s.signum()
    } else {
        s
    }
}

/// Cubic Hermite interpolant between p1..p2 with endpoint tangents m1, m2.
#[inline]
fn cubic_hermite(p1: f32, p2: f32, m1: f32, m2: f32, t: f32) -> f32 {
    let t2 = t * t; let t3 = t2 * t;
    (2.0*t3 - 3.0*t2 + 1.0)*p1 + (t3 - 2.0*t2 + t)*m1
    + (-2.0*t3 + 3.0*t2)*p2 + (t3 - t2)*m2
}

/// Akima tangent at a node, given four consecutive finite differences.
/// d_prev2, d_prev1 are the two differences to the left; d_curr, d_next to the right.
#[inline]
fn akima_tangent(d_prev2: f32, d_prev1: f32, d_curr: f32, d_next: f32) -> f32 {
    let w1 = (d_next  - d_curr ).abs();
    let w2 = (d_prev1 - d_prev2).abs();
    if w1 + w2 < 1e-10 {
        (d_prev1 + d_curr) * 0.5
    } else {
        (w1 * d_prev1 + w2 * d_curr) / (w1 + w2)
    }
}

/// Unified sub-bin dispatcher.  `center` is a fractional bin index into `norms`.
fn interp_sub_bin(norms: &[f32], center: f32, mode: &InterpolationMode) -> f32 {
    let half = norms.len();
    if half == 0 { return 0.0 }
    // Every neighbour below is clamped into range; `b1` was not, which made
    // this function safe for the bins around the one it actually reads and
    // unsafe for that one. It is the index that is always used, so it is the
    // one that has to hold on its own.
    let b1 = (center.floor().max(0.0) as usize).min(half - 1);
    let t  = (center - b1 as f32).clamp(0.0, 1.0);
    match mode {
        InterpolationMode::None => {
            norms[b1] // nearest bin — no interpolation
        }
        InterpolationMode::Linear => {
            let b2 = (b1 + 1).min(half - 1);
            (norms[b1] * (1.0 - t) + norms[b2] * t).max(0.0)
        }
        InterpolationMode::CatmullRom => {
            let b0 = b1.saturating_sub(1);
            let b2 = (b1 + 1).min(half - 1);
            let b3 = (b1 + 2).min(half - 1);
            catmull_rom(norms[b0], norms[b1], norms[b2], norms[b3], t).max(0.0)
        }
        InterpolationMode::Pchip => {
            let b0 = b1.saturating_sub(1);
            let b2 = (b1 + 1).min(half - 1);
            let b3 = (b1 + 2).min(half - 1);
            let d0 = norms[b1] - norms[b0];
            let d1 = norms[b2] - norms[b1];
            let d2 = norms[b3] - norms[b2];
            let m1 = pchip_slope(d0, d1);
            let m2 = pchip_slope(d1, d2);
            cubic_hermite(norms[b1], norms[b2], m1, m2, t).max(0.0)
        }
        InterpolationMode::Akima => {
            let bm2 = b1.saturating_sub(2);
            let bm1 = b1.saturating_sub(1);
            let bp1 = (b1 + 1).min(half - 1);
            let bp2 = (b1 + 2).min(half - 1);
            let bp3 = (b1 + 3).min(half - 1);
            // 5 finite differences spanning bm2..bp3
            let d0 = norms[bm1] - norms[bm2];
            let d1 = norms[b1]  - norms[bm1];
            let d2 = norms[bp1] - norms[b1];
            let d3 = norms[bp2] - norms[bp1];
            let d4 = norms[bp3] - norms[bp2];
            let m1 = akima_tangent(d0, d1, d2, d3);
            let m2 = akima_tangent(d1, d2, d3, d4);
            cubic_hermite(norms[b1], norms[bp1], m1, m2, t).max(0.0)
        }
        InterpolationMode::Lanczos => {
            const A: usize = 3;
            let b1i = b1 as isize;
            let mut sum = 0.0f32;
            let mut weight = 0.0f32;
            for k in (b1i - A as isize + 1)..=(b1i + A as isize) {
                let b = k.clamp(0, half as isize - 1) as usize;
                let x = center - k as f32;
                let w = lanczos_kernel(x, A as f32);
                sum += norms[b] * w;
                weight += w;
            }
            if weight > 0.0 { (sum / weight).max(0.0) } else { 0.0 }
        }
    }
}

/// Colour ramp for the spectrum visualisers. `Classic` is the original rainbow
/// heat map; the others are cohesive alternatives, and `Accent` follows the
/// current track's album-art accent so the graph matches the surrounding chrome.
/// Selectable (and persisted) from the app's Appearance menu.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum SpectrumPalette {
    Classic,
    Kugelblitz,
    Ice,
    Magma,
    Aurora,
    Mono,
    Accent,
}

impl Default for SpectrumPalette {
    fn default() -> Self { SpectrumPalette::Classic }
}

impl SpectrumPalette {
    /// All variants, in menu order.
    pub const ALL: [SpectrumPalette; 7] = [
        SpectrumPalette::Classic,
        SpectrumPalette::Kugelblitz,
        SpectrumPalette::Ice,
        SpectrumPalette::Magma,
        SpectrumPalette::Aurora,
        SpectrumPalette::Mono,
        SpectrumPalette::Accent,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SpectrumPalette::Classic    => "Classic",
            SpectrumPalette::Kugelblitz => "Kugelblitz",
            SpectrumPalette::Ice        => "Ice",
            SpectrumPalette::Magma      => "Magma",
            SpectrumPalette::Aurora     => "Aurora",
            SpectrumPalette::Mono       => "Mono",
            SpectrumPalette::Accent     => "Album accent",
        }
    }
}

/// A resolved palette: the chosen ramp plus the accent it may need (used only
/// by `SpectrumPalette::Accent`). Small and `Copy`, so passed by value/ref freely.
#[derive(Clone, Copy)]
pub struct Palette {
    pub kind:   SpectrumPalette,
    pub accent: Color32,
}

impl Palette {
    pub fn new(kind: SpectrumPalette, accent: Color32) -> Self { Self { kind, accent } }

    /// Colour for a normalised bar / point magnitude (0–1).
    pub fn bar(&self, t: f32) -> Color32 { self.sample(t) }
    /// Colour for a spectrogram / waterfall cell (0–1).
    pub fn heat(&self, t: f32) -> Color32 { self.sample(t) }
    /// A representative "hot" colour, for line / filled outline styles.
    pub fn line(&self) -> Color32 { self.sample(0.82) }

    fn sample(&self, t: f32) -> Color32 {
        let t = t.clamp(0.0, 1.0);
        match self.kind {
            SpectrumPalette::Classic => sample_stops(&[
                (0.0, (0, 0, 0)), (0.2, (0, 0, 140)), (0.4, (0, 220, 255)),
                (0.6, (255, 220, 0)), (0.8, (255, 70, 0)), (1.0, (255, 255, 220)),
            ], t),
            SpectrumPalette::Kugelblitz => sample_stops(&[
                (0.0, (10, 12, 24)), (0.45, (60, 90, 200)),
                (0.78, (148, 177, 255)), (1.0, (232, 240, 255)),
            ], t),
            SpectrumPalette::Ice => sample_stops(&[
                (0.0, (2, 6, 20)), (0.35, (0, 70, 150)),
                (0.7, (0, 190, 220)), (1.0, (224, 255, 255)),
            ], t),
            SpectrumPalette::Magma => sample_stops(&[
                (0.0, (2, 0, 6)), (0.25, (70, 16, 96)), (0.5, (160, 44, 96)),
                (0.75, (240, 110, 60)), (1.0, (255, 246, 200)),
            ], t),
            SpectrumPalette::Aurora => sample_stops(&[
                (0.0, (2, 12, 10)), (0.35, (0, 96, 76)),
                (0.66, (44, 200, 126)), (1.0, (204, 255, 190)),
            ], t),
            SpectrumPalette::Mono => sample_stops(&[
                (0.0, (8, 8, 10)), (1.0, (240, 242, 248)),
            ], t),
            SpectrumPalette::Accent => {
                let a = self.accent;
                // A low tint of the accent (keeps a little hue out of the floor),
                // the accent itself in the body, pushed toward white at the peak.
                let lo = (a.r() / 6, a.g() / 6, (a.b() / 5).max(14));
                let mid = (a.r(), a.g(), a.b());
                let hi = (
                    (a.r() as u16 + 170).min(255) as u8,
                    (a.g() as u16 + 170).min(255) as u8,
                    (a.b() as u16 + 170).min(255) as u8,
                );
                sample_stops(&[(0.0, (6, 7, 12)), (0.28, lo), (0.7, mid), (1.0, hi)], t)
            }
        }
    }
}

/// Linearly interpolate a colour from an ascending list of `(pos, rgb)` stops.
fn sample_stops(stops: &[(f32, (u8, u8, u8))], t: f32) -> Color32 {
    let rgb = |c: (u8, u8, u8)| Color32::from_rgb(c.0, c.1, c.2);
    match stops.first() {
        None => return Color32::BLACK,
        Some(&(p0, c0)) if t <= p0 => return rgb(c0),
        _ => {}
    }
    let &(pl, cl) = stops.last().unwrap();
    if t >= pl { return rgb(cl); }
    for w in stops.windows(2) {
        let (p0, c0) = w[0];
        let (p1, c1) = w[1];
        if t >= p0 && t <= p1 {
            let s = if (p1 - p0).abs() < 1e-6 { 0.0 } else { (t - p0) / (p1 - p0) };
            let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * s).round() as u8;
            return Color32::from_rgb(
                lerp(c0.0, c1.0), lerp(c0.1, c1.1), lerp(c0.2, c1.2),
            );
        }
    }
    rgb(cl)
}

/// Paint a small left→right gradient preview of a palette (its low-to-high
/// magnitude ramp), used as a swatch in the palette picker.
fn paint_palette_swatch(ui: &mut egui::Ui, pal: &Palette, size: egui::Vec2) {
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    if !ui.is_rect_visible(rect) { return; }
    let painter = ui.painter();
    let n = 22usize;
    for i in 0..n {
        let t  = i as f32 / (n - 1) as f32;
        let x0 = rect.left() + rect.width() * (i as f32) / (n as f32);
        let x1 = rect.left() + rect.width() * ((i + 1) as f32) / (n as f32);
        painter.rect_filled(
            Rect::from_min_max(Pos2::new(x0, rect.top()), Pos2::new(x1, rect.bottom())),
            0.0, pal.bar(t),
        );
    }
}

// Theme-aware text colours for the spectrum window's chrome labels (the window
// itself follows the app's light/dark theme). Plot-painted text keeps its own
// fixed colours since the plot area stays dark in both themes.
fn txt_dim(dark: bool) -> Color32 {
    if dark { Color32::from_gray(150) } else { Color32::from_rgb(0x55, 0x58, 0x63) }
}
fn txt_faint(dark: bool) -> Color32 {
    if dark { Color32::from_gray(120) } else { Color32::from_rgb(0x74, 0x77, 0x82) }
}
fn txt_ok(dark: bool) -> Color32 {
    if dark { Color32::from_rgb(100, 210, 100) } else { Color32::from_rgb(0x1c, 0x77, 0x30) }
}
fn txt_warn(dark: bool) -> Color32 {
    if dark { Color32::from_rgb(255, 200, 60) } else { Color32::from_rgb(0x93, 0x63, 0x00) }
}
fn txt_accent(dark: bool) -> Color32 {
    if dark { Color32::from_rgb(140, 190, 255) } else { Color32::from_rgb(0x2f, 0x50, 0xc0) }
}

// ---------------------------------------------------------------------------
// EQ overlay — response curve and draggable band nodes
// ---------------------------------------------------------------------------

/// EQ dB range shown on the overlay (±DB_RANGE maps to ±half plot height).
const EQ_DB_RANGE: f32 = 24.0;

fn eq_node_pos(
    band: &EqBand, plot_rect: Rect, min_freq: f32, max_freq: f32,
    scale: freq_scale::FreqScale,
) -> egui::Pos2 {
    // Positioned on the same axis the bars are drawn on. On the ERB scale a
    // node placed by the log formula would sit more than an octave away from
    // the band it represents down in the bass.
    let t = scale.position_of(band.freq, min_freq, max_freq).clamp(0.0, 1.0);
    let x = plot_rect.left() + t * plot_rect.width();
    let gain = if band.kind.has_gain() { band.gain_db } else { 0.0 };
    let y_center = plot_rect.center().y;
    let y_per_db = (plot_rect.height() * 0.5) / EQ_DB_RANGE;
    let y = (y_center - gain * y_per_db).clamp(plot_rect.top(), plot_rect.bottom());
    egui::Pos2::new(x, y)
}

fn draw_eq_overlay(
    painter: &egui::Painter,
    bands: &[EqBand],
    sr: u32,
    plot_rect: Rect,
    min_freq: f32,
    max_freq: f32,
    draw_curve: bool,
    hovered: Option<usize>,
    dragging: Option<usize>,
    scale: freq_scale::FreqScale,
) {
    use egui::{Color32, Pos2, Stroke};

    let y_center = plot_rect.center().y;
    let y_per_db = (plot_rect.height() * 0.5) / EQ_DB_RANGE;
    let w = plot_rect.width() as usize;

    if draw_curve && !bands.is_empty() {
        // 0 dB reference line
        painter.line_segment(
            [Pos2::new(plot_rect.left(), y_center), Pos2::new(plot_rect.right(), y_center)],
            Stroke::new(0.5, Color32::from_rgba_unmultiplied(255, 255, 255, 30)),
        );

        // Individual band curves (dimmed)
        for (i, band) in bands.iter().enumerate() {
            if !band.enabled { continue; }
            let col = eq_band_color(i).linear_multiply(0.35);
            let pts: Vec<Pos2> = (0..=w).map(|px| {
                let t = px as f32 / w as f32;
                let freq = scale.freq_at(t, min_freq, max_freq);
                let (bv, av) = eq_biquad_coeffs(band, sr);
                let db = biquad_response_db(bv, av, freq, sr).clamp(-EQ_DB_RANGE, EQ_DB_RANGE);
                Pos2::new(plot_rect.left() + t * plot_rect.width(),
                          (y_center - db * y_per_db).clamp(plot_rect.top(), plot_rect.bottom()))
            }).collect();
            for seg in pts.windows(2) {
                painter.line_segment([seg[0], seg[1]], Stroke::new(1.0, col));
            }
        }

        // Combined response curve
        let pts: Vec<Pos2> = (0..=w).map(|px| {
            let t = px as f32 / w as f32;
            let freq = scale.freq_at(t, min_freq, max_freq);
            let db = total_eq_response_db(bands, sr, freq).clamp(-EQ_DB_RANGE, EQ_DB_RANGE);
            Pos2::new(plot_rect.left() + t * plot_rect.width(),
                      (y_center - db * y_per_db).clamp(plot_rect.top(), plot_rect.bottom()))
        }).collect();
        for seg in pts.windows(2) {
            painter.line_segment([seg[0], seg[1]],
                Stroke::new(2.0, Color32::from_rgba_unmultiplied(255, 220, 80, 220)));
        }
    }

    // Band nodes
    for (i, band) in bands.iter().enumerate() {
        if !band.enabled { continue; }
        let pos = eq_node_pos(band, plot_rect, min_freq, max_freq, scale);
        let is_active = hovered == Some(i) || dragging == Some(i);
        let r = if is_active { 9.0 } else { 6.0 };
        let col = eq_band_color(i);
        painter.circle_filled(pos, r, col.linear_multiply(if is_active { 1.0 } else { 0.75 }));
        painter.circle_stroke(pos, r, Stroke::new(1.5, Color32::WHITE));
        // Small index label
        painter.text(pos, egui::Align2::CENTER_CENTER,
            format!("{}", i + 1),
            egui::FontId::monospace(8.0), Color32::BLACK);
    }
}

fn draw_bars(painter: &egui::Painter, mags: &[f32], rect: Rect, gap: f32, pal: &Palette) {
    let n = mags.len();
    if n == 0 { return; }
    let ppp       = painter.ctx().pixels_per_point();
    let phys_left = (rect.left()  * ppp).round() as i32;
    let phys_w    = ((rect.right() * ppp).round() as i32 - phys_left).max(1);
    let phys_gap  = (gap * ppp).round().max(0.0) as i32;

    // Each column needs (1 + phys_gap) physical pixels to show a visible gap.
    let min_col_w = (1 + phys_gap).max(1);
    let draw_n    = ((phys_w / min_col_w) as usize).min(n).max(1);

    // Build a single mesh for all bars — one painter.add() call instead of
    // draw_n separate rect_filled() calls.  This eliminates the O(N) per-frame
    // CPU cost that was causing periodic audio interruptions proportional to
    // bar count (more bars → longer render → longer audio glitch → lower pitch).
    let mut mesh = egui::Mesh::default();
    mesh.reserve_triangles(draw_n * 2);
    mesh.reserve_vertices(draw_n * 4);

    for i in 0..draw_n {
        let src_lo = (i * n) / draw_n;
        let src_hi = (((i + 1) * n) / draw_n).min(n);
        let v = mags[src_lo..src_hi].iter().cloned().fold(0.0_f32, f32::max);

        let h   = v * rect.height();
        let px0 = phys_left + (i as i32 * phys_w) / draw_n as i32;
        let px1 = (phys_left + ((i + 1) as i32 * phys_w) / draw_n as i32 - phys_gap).max(px0 + 1);

        let x0 = px0 as f32 / ppp;
        let x1 = px1 as f32 / ppp;
        let y0 = rect.bottom() - h;
        let y1 = rect.bottom();
        let c  = pal.bar(v);
        let base = mesh.vertices.len() as u32;
        mesh.colored_vertex(Pos2::new(x0, y0), c);  // top-left
        mesh.colored_vertex(Pos2::new(x1, y0), c);  // top-right
        mesh.colored_vertex(Pos2::new(x1, y1), c);  // bottom-right
        mesh.colored_vertex(Pos2::new(x0, y1), c);  // bottom-left
        mesh.indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }
    painter.add(Shape::mesh(mesh));
}

/// Right minus left, per bar, about a centre line.
///
/// The difference axis runs across the plot and the frequency axis along it;
/// which is which, which way round each goes, and which channel sits on the
/// positive end are all [`channels::DiffLayout`], because none of them has a
/// correct answer. The axis is
/// [`channels::DIFF_FULL_SCALE_DB`] either side, not the plot's usual 80 dB:
/// channel differences worth looking at are single figures, and on an 80 dB
/// axis every recording ever made is a flat line.
///
/// Bars are folded to the drawable column count exactly as the other renderers
/// do, but by the *largest magnitude* in each group rather than the maximum
/// value — the extreme of a group that is 6 dB left is −6, and taking a maximum
/// would report it as whatever the least-left bar in the group happened to be.
fn draw_channel_diff(
    painter: &egui::Painter,
    left: &[f32],
    right: &[f32],
    rect: Rect,
    gap: f32,
    layout: channels::DiffLayout,
) {
    let n = left.len().min(right.len());
    if n == 0 {
        return;
    }
    let vertical = layout.orientation == channels::DiffOrientation::Vertical;

    // One geometry, two orientations. `along` is the frequency axis and
    // `across` the difference axis; everything below is written in those terms
    // and mapped to x/y once, at the end, rather than duplicated.
    let ppp = painter.ctx().pixels_per_point();
    let (along_min, along_max, across_mid, across_half) = if vertical {
        (
            rect.top(),
            rect.bottom(),
            rect.center().x,
            rect.width() * 0.5,
        )
    } else {
        (
            rect.left(),
            rect.right(),
            rect.center().y,
            rect.height() * 0.5,
        )
    };

    let phys_start = (along_min * ppp).round() as i32;
    let phys_span = ((along_max * ppp).round() as i32 - phys_start).max(1);
    let phys_gap = (gap * ppp).round().max(0.0) as i32;
    let min_col = (1 + phys_gap).max(1);
    let draw_n = ((phys_span / min_col) as usize).min(n).max(1);

    // The zero line and every other piece of axis furniture belong to
    // `draw_diff_axes`, which runs after this and therefore draws over the
    // bars — the same order the ordinary plot uses for its gridlines.

    let mut mesh = egui::Mesh::default();
    mesh.reserve_triangles(draw_n * 2);
    mesh.reserve_vertices(draw_n * 4);

    for i in 0..draw_n {
        // The frequency flip is applied to the *display* position, so the
        // grouping below still folds neighbouring frequencies together.
        let slot = layout.source_bar(i, draw_n);
        let lo = (slot * n) / draw_n;
        let hi = (((slot + 1) * n) / draw_n).min(n).max(lo + 1);
        let mut raw = 0.0f32;
        for b in lo..hi {
            let f = channels::diff_fraction(left[b], right[b]);
            if f.abs() > raw.abs() {
                raw = f;
            }
        }
        // Colour is decided before the flip and position after it. Swapping the
        // sides moves a band across the plot; it must not repaint it, or the
        // key stops meaning anything.
        let who = channels::louder(raw);
        let d = layout.placed(raw);

        let p0 = phys_start + (i as i32 * phys_span) / draw_n as i32;
        let p1 = (phys_start + ((i + 1) as i32 * phys_span) / draw_n as i32 - phys_gap)
            .max(p0 + 1);
        let a0 = p0 as f32 / ppp;
        let a1 = p1 as f32 / ppp;

        // At least one pixel, so a bar that is nearly zero is still visibly a
        // bar sitting on the line rather than nothing at all.
        let ext = (d.abs() * across_half).max(1.0 / ppp);
        // Positive is up in the horizontal form and right in the vertical one:
        // screen y grows downward, so the horizontal case subtracts.
        let (c0, c1) = if d >= 0.0 {
            if vertical {
                (across_mid, across_mid + ext)
            } else {
                (across_mid - ext, across_mid)
            }
        } else if vertical {
            (across_mid - ext, across_mid)
        } else {
            (across_mid, across_mid + ext)
        };
        let c = channel_color(who);

        let quad = if vertical {
            [
                Pos2::new(c0, a0),
                Pos2::new(c1, a0),
                Pos2::new(c1, a1),
                Pos2::new(c0, a1),
            ]
        } else {
            [
                Pos2::new(a0, c0),
                Pos2::new(a1, c0),
                Pos2::new(a1, c1),
                Pos2::new(a0, c1),
            ]
        };
        let base = mesh.vertices.len() as u32;
        for pos in quad {
            mesh.colored_vertex(pos, c);
        }
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    painter.add(Shape::mesh(mesh));

}

/// The colour that names a channel, wherever it is drawn.
///
/// One place, so a plot and its key cannot disagree. `Equal` is deliberately
/// neutral rather than either channel: a band with no difference belongs to
/// neither, and painting it cyan would read as "left, quietly".
fn channel_color(who: channels::Louder) -> Color32 {
    match who {
        channels::Louder::Left => CHANNEL_L_COLOR,
        channels::Louder::Right => CHANNEL_R_COLOR,
        channels::Louder::Equal => Color32::from_gray(90),
    }
}

/// Frequencies worth a tick, shared by the ordinary plot and the Diff view so
/// switching between them does not move the grid.
const FREQ_TICKS: &[f32] = &[
    50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0,
];

/// Draw a label anchored at `at`, nudged so the whole of it stays inside the
/// painter's clip rectangle. Returns the box it occupies.
///
/// Text is anchored at a point and grows around it, so a centred label whose
/// anchor sits on an edge puts half of itself outside the clip and the reader
/// sees a truncated number. On a difference axis the two endpoints are *always*
/// on an edge — that is what makes them endpoints — and the production painter
/// is `ui.painter_at(rect)` with the plot reaching that rectangle's top and
/// right sides, so there is nothing outside to spill into.
///
/// The galley is laid out first and positioned second, rather than picking an
/// alignment and hoping: the nudge needs the text's real size, which differs
/// between "0" and "R +20" and between the two fonts in use here.
fn text_inside(
    painter: &egui::Painter,
    at: Pos2,
    align: egui::Align2,
    text: String,
    font: egui::FontId,
    color: Color32,
) -> Rect {
    let galley = painter.layout_no_wrap(text, font, color);
    let size = galley.size();
    let mut pos = align
        .align_size_within_rect(size, Rect::from_min_max(at, at))
        .min;
    let clip = painter.clip_rect();
    // `max` before `clamp`, so a clip narrower than the text pins the label to
    // the near edge instead of panicking on an inverted range.
    pos.x = pos.x.clamp(clip.left(), (clip.right() - size.x).max(clip.left()));
    pos.y = pos.y.clamp(clip.top(), (clip.bottom() - size.y).max(clip.top()));
    painter.galley(pos, galley, color);
    Rect::from_min_size(pos, size)
}

/// Ticks eligible for `sample_rate`, in ascending order, as (frequency, position).
///
/// Two different bounds, and conflating them is what put the labels on the wrong
/// bars:
///
/// * **Eligibility** is Nyquist and the configured ceiling. A tick above either
///   names a frequency the plot cannot be showing.
/// * **Position** is the configured `min_freq..max_freq`, because that is the
///   range the *bars* are laid out over — `bar_center` and the analyser's own
///   `freq_at` both use it, and bins above Nyquist are clamped by index rather
///   than by rescaling the axis.
///
/// Using Nyquist to place the ticks made the two coordinate systems disagree
/// whenever `max_freq` exceeded it: at 44.1 kHz with the default 24 kHz ceiling
/// every label sat about 4 % of the width to the right of the bar it named.
///
/// Ticks below `min_freq` are dropped rather than clamped. Clamping piled them
/// all onto the left endpoint, which reads as a real tick at a frequency that
/// is not on the axis.
fn eligible_ticks(sample_rate: u32, min_freq: f32, max_freq: f32,
                  scale: freq_scale::FreqScale) -> Vec<(f32, f32)> {
    let nyquist = sample_rate as f32 / 2.0;
    let ceiling = nyquist.min(max_freq);
    if !(min_freq < ceiling) {
        return Vec::new();
    }
    FREQ_TICKS
        .iter()
        .copied()
        .filter(|&f| f >= min_freq && f <= ceiling)
        .map(|f| (f, scale.position_of(f, min_freq, max_freq)))
        .filter(|&(_, t)| (0.0..=1.0).contains(&t))
        .collect()
}

/// Label for a tick frequency.
fn tick_label(freq: f32) -> String {
    if freq >= 1000.0 {
        format!("{}k", (freq / 1000.0) as u32)
    } else {
        format!("{}", freq as u32)
    }
}

/// Whether the plot on screen is a Diff plot, and so needs [`draw_diff_axes`]
/// rather than the ordinary pair.
///
/// Takes the *effective* view and whether the channel renderer actually drew,
/// never the requested view. Both inputs matter and for different reasons: a
/// Diff selection with no stereo tap resolves to Mix through
/// `channels::effective_view` and must get ordinary axes, and a Diff selection
/// that resolved fine can still fail at the last moment if the frame handed to
/// the renderer is missing a channel. Branching on the selection alone drew
/// difference axes over an ordinary spectrum.
fn uses_diff_axes(effective: channels::ChannelView, drew_channels: bool) -> bool {
    drew_channels && effective == channels::ChannelView::Diff
}

/// Axes for the Diff view.
///
/// The ordinary [`draw_db_labels`] / [`draw_freq_labels`] pair cannot serve this
/// plot, and drawing them anyway is what left a −70 dB gridline lying across a
/// plot whose centre is 0 and a frequency axis along the edge that was showing
/// decibels. Two things are different here:
///
/// * the difference axis is ±[`channels::DIFF_FULL_SCALE_DB`] about a zero in
///   the middle, not 0…−80 dB of level rising from the bottom;
/// * in the vertical arrangement the two axes are swapped outright, so
///   frequency runs down the side and the difference runs across.
///
/// Both flips are honoured: `flip_channels` decides which channel each end
/// belongs to (and therefore the colour and letter there), and `flip_frequency`
/// reverses the tick positions through the same
/// [`channels::DiffLayout::along_fraction`] the bars are placed with.
///
/// Runs *after* the bars, like the ordinary gridlines, so the zero line stays
/// visible across a bar that sits on it.
fn draw_diff_axes(
    painter: &egui::Painter,
    rect: Rect,
    layout: channels::DiffLayout,
    sample_rate: u32,
    min_freq: f32,
    max_freq: f32,
    scale: freq_scale::FreqScale,
) {
    let vertical = layout.orientation == channels::DiffOrientation::Vertical;
    let full = channels::DIFF_FULL_SCALE_DB;
    let font = egui::FontId::monospace(9.0);
    let end_font = egui::FontId::proportional(11.0);

    let (across_mid, across_half) = if vertical {
        (rect.center().x, rect.width() * 0.5)
    } else {
        (rect.center().y, rect.height() * 0.5)
    };
    let (along_min, along_span) = if vertical {
        (rect.top(), rect.height())
    } else {
        (rect.left(), rect.width())
    };

    // Positive is up in the horizontal form — screen y grows downward, so it
    // subtracts — and to the right in the vertical one.
    let across_at = |db: f32| -> f32 {
        let f = (db / full).clamp(-1.0, 1.0);
        if vertical {
            across_mid + f * across_half
        } else {
            across_mid - f * across_half
        }
    };

    // ── difference axis ────────────────────────────────────────────────────
    let (pos_ch, neg_ch) = layout.end_channels();
    for db in [-full, -full * 0.5, 0.0, full * 0.5, full] {
        let c = across_at(db);
        let zero = db == 0.0;
        let seg = if vertical {
            [Pos2::new(c, rect.top()), Pos2::new(c, rect.bottom())]
        } else {
            [Pos2::new(rect.left(), c), Pos2::new(rect.right(), c)]
        };
        painter.line_segment(
            seg,
            Stroke::new(
                if zero { 1.0 } else { 0.5 },
                Color32::from_gray(if zero { 70 } else { 34 }),
            ),
        );

        // The ends carry the channel they belong to, in that channel's colour;
        // everything between is a plain signed number.
        let (text, color, f) = if db >= full {
            (format!("{} +{full:.0}", pos_ch.label()), channel_color(pos_ch), end_font.clone())
        } else if db <= -full {
            (format!("{} +{full:.0}", neg_ch.label()), channel_color(neg_ch), end_font.clone())
        } else if zero {
            ("0".to_string(), Color32::from_gray(80), font.clone())
        } else {
            (format!("{db:+.0}"), Color32::from_gray(80), font.clone())
        };
        // The difference axis takes whichever margin frequency is not using.
        let (at, align) = if vertical {
            (Pos2::new(c, rect.bottom() + 5.0), egui::Align2::CENTER_TOP)
        } else {
            (Pos2::new(rect.left() - 4.0, c), egui::Align2::RIGHT_CENTER)
        };
        text_inside(painter, at, align, text, f, color);
    }

    // ── frequency axis ─────────────────────────────────────────────────────
    for (freq, t) in eligible_ticks(sample_rate, min_freq, max_freq, scale) {
        let a = along_min + layout.along_fraction(t) * along_span;
        let (tick, at, align) = if vertical {
            (
                [Pos2::new(rect.left() - 4.0, a), Pos2::new(rect.left(), a)],
                Pos2::new(rect.left() - 6.0, a),
                egui::Align2::RIGHT_CENTER,
            )
        } else {
            (
                [
                    Pos2::new(a, rect.bottom()),
                    Pos2::new(a, rect.bottom() + 4.0),
                ],
                Pos2::new(a, rect.bottom() + 5.0),
                egui::Align2::CENTER_TOP,
            )
        };
        painter.line_segment(tick, Stroke::new(1.0, Color32::from_gray(60)));
        text_inside(
            painter,
            at,
            align,
            tick_label(freq),
            font.clone(),
            Color32::from_gray(90),
        );
    }
}

fn draw_peak_hold(
    painter:  &egui::Painter,
    peaks:    &[f32],
    alphas:   &[f32],
    rect:     Rect,
    gap:      f32,
    cfg:      &PeakHoldConfig,
) {
    let n = peaks.len();
    if n == 0 { return; }
    let ppp           = painter.ctx().pixels_per_point();
    let phys_left     = (rect.left()  * ppp).round() as i32;
    let phys_w        = ((rect.right() * ppp).round() as i32 - phys_left).max(1);
    let phys_gap      = (gap * ppp).round().max(0.0) as i32;
    let min_col_w     = (1 + phys_gap).max(1);
    let draw_n        = ((phys_w / min_col_w) as usize).min(n).max(1);
    let phys_thick    = (cfg.peak_thickness as i32).max(1);
    let phys_rect_top = (rect.top()    * ppp).round() as i32;

    let base_color = cfg.color;
    let mut mesh = egui::Mesh::default();
    mesh.reserve_triangles(draw_n * 2);
    mesh.reserve_vertices(draw_n * 4);

    for i in 0..draw_n {
        let src_lo = (i * n) / draw_n;
        let src_hi = (((i + 1) * n) / draw_n).min(n);
        let v     = peaks[src_lo..src_hi].iter().cloned().fold(0.0_f32, f32::max).clamp(0.0, 1.0);
        let alpha = alphas[src_lo..src_hi].iter().cloned().fold(0.0_f32, f32::max);
        if v <= 0.0 || alpha <= 0.0 { continue; }

        let px0 = phys_left + (i as i32 * phys_w) / draw_n as i32;
        let px1 = (phys_left + ((i + 1) as i32 * phys_w) / draw_n as i32 - phys_gap).max(px0 + 1);

        // Bottom edge of marker = top of bar; clamp so marker stays inside plot rect
        let phys_bar_top = ((rect.bottom() * ppp).round() as i32
            - (v * rect.height() * ppp).round() as i32)
            .max(phys_rect_top);
        let py1 = phys_bar_top;
        let py0 = (py1 - phys_thick).max(phys_rect_top);
        if py0 >= py1 { continue; }

        let x0 = px0 as f32 / ppp;
        let x1 = px1 as f32 / ppp;
        let y0 = py0 as f32 / ppp;
        let y1 = py1 as f32 / ppp;

        let a = (alpha * base_color.a() as f32) as u8;
        let color = Color32::from_rgba_unmultiplied(base_color.r(), base_color.g(), base_color.b(), a);
        let base = mesh.vertices.len() as u32;
        mesh.colored_vertex(Pos2::new(x0, y0), color);
        mesh.colored_vertex(Pos2::new(x1, y0), color);
        mesh.colored_vertex(Pos2::new(x1, y1), color);
        mesh.colored_vertex(Pos2::new(x0, y1), color);
        mesh.indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }
    painter.add(Shape::mesh(mesh));
}

fn draw_line(painter: &egui::Painter, mags: &[f32], rect: Rect, color: Color32) {
    let n = mags.len();
    if n < 2 { return; }
    let points: Vec<Pos2> = mags.iter().enumerate().map(|(i, &v)| {
        let x = rect.left() + (i as f32 / (n - 1) as f32) * rect.width();
        let y = rect.bottom() - v * rect.height();
        Pos2::new(x, y)
    }).collect();
    painter.add(Shape::line(points, Stroke::new(1.5, color)));
}

fn draw_filled(painter: &egui::Painter, mags: &[f32], rect: Rect, pal: &Palette) {
    let n = mags.len();
    if n < 2 { return; }
    let line_color = pal.line();
    let mid        = pal.bar(0.5);
    let fill_color = Color32::from_rgba_unmultiplied(mid.r(), mid.g(), mid.b(), 110);

    // Build a quad mesh (two triangles per adjacent pair of spectrum points).
    // This correctly fills non-convex shapes; PathShape tessellation does not.
    let mut mesh = egui::Mesh::default();
    for i in 0..n - 1 {
        let x0 = rect.left() + (i as f32 / (n - 1) as f32) * rect.width();
        let x1 = rect.left() + ((i + 1) as f32 / (n - 1) as f32) * rect.width();
        let y0 = rect.bottom() - mags[i] * rect.height();
        let y1 = rect.bottom() - mags[i + 1] * rect.height();
        let base = mesh.vertices.len() as u32;
        mesh.colored_vertex(Pos2::new(x0, y0), fill_color);            // 0 top-left
        mesh.colored_vertex(Pos2::new(x1, y1), fill_color);            // 1 top-right
        mesh.colored_vertex(Pos2::new(x1, rect.bottom()), fill_color); // 2 bottom-right
        mesh.colored_vertex(Pos2::new(x0, rect.bottom()), fill_color); // 3 bottom-left
        mesh.indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }
    painter.add(Shape::mesh(mesh));
    draw_line(painter, mags, rect, line_color);
}

/// How to cut the waterfall ring into two quads so it reads in order on screen.
///
/// `head` is where the *next* row will be written; because rows are written
/// backwards, the newest is at `head + 1`. Screen top-to-bottom is therefore
/// ascending texture rows starting there, which wraps exactly once.
///
/// Returns the screen fraction at which the wrap falls, and the texture
/// v-ranges of the top and bottom quads.
fn waterfall_slices(head: usize, h: usize) -> (f32, (f32, f32), (f32, f32)) {
    let start = (head + 1) % h.max(1);
    let v = start as f32 / h as f32;
    // The top quad shows [start, h), which is (h - start) of h rows, so it must
    // occupy exactly that fraction of the height or the rows would be stretched.
    ((h - start) as f32 / h as f32, (v, 1.0), (0.0, v))
}

/// One waterfall row as a 1-pixel-tall image, for a partial texture upload.
fn waterfall_row_image(row: &[f32], n: usize, pal: &Palette) -> egui::ColorImage {
    let mut pixels = vec![Color32::BLACK; n];
    for (col, px) in pixels.iter_mut().enumerate().take(row.len().min(n)) {
        *px = pal.heat(row[col]);
    }
    egui::ColorImage { size: [n, 1], pixels }
}

fn draw_phasescope(painter: &egui::Painter, frames: &[[f32; 2]], rect: Rect, correlation: f32) {
    // Dark background
    painter.rect_filled(rect, 0.0, Color32::from_rgb(5, 8, 14));
    let n = frames.len().min(4096);
    if n == 0 { return; }
    let start = frames.len().saturating_sub(n);
    let cx = rect.center().x;
    let cy = rect.center().y;
    let scale = rect.width().min(rect.height()) * 0.45;
    // Centre cross-hairs
    painter.line_segment([Pos2::new(rect.left(), cy), Pos2::new(rect.right(), cy)],
        Stroke::new(0.5, Color32::from_gray(30)));
    painter.line_segment([Pos2::new(cx, rect.top()), Pos2::new(cx, rect.bottom())],
        Stroke::new(0.5, Color32::from_gray(30)));
    // Diagonal reference (mono line)
    painter.line_segment(
        [Pos2::new(cx - scale * 0.7, cy + scale * 0.7), Pos2::new(cx + scale * 0.7, cy - scale * 0.7)],
        Stroke::new(0.5, Color32::from_gray(25)));
    // Points: M/S rotation — mono on horizontal, side on vertical
    let sq2 = std::f32::consts::SQRT_2;
    for (i, &[l, r]) in frames[start..].iter().enumerate() {
        let mx = (l + r) / sq2;
        let sy = (l - r) / sq2;
        let px = cx + mx * scale;
        let py = cy - sy * scale;
        if !rect.contains(Pos2::new(px, py)) { continue; }
        let alpha = (i as f32 / n as f32 * 200.0 + 30.0) as u8;
        painter.circle_filled(Pos2::new(px, py), 1.2, Color32::from_rgba_unmultiplied(80, 220, 120, alpha));
    }
    // Correlation readout
    let corr_str = format!("corr {:.2}", correlation);
    let color = if correlation > 0.3 { Color32::from_gray(140) }
                else if correlation > -0.2 { Color32::from_rgb(220, 180, 60) }
                else { Color32::from_rgb(220, 80, 80) };
    painter.text(Pos2::new(rect.left() + 4.0, rect.top() + 4.0),
        egui::Align2::LEFT_TOP, corr_str, egui::FontId::monospace(10.0), color);
}

const DB_MARGIN: f32   = 34.0; // px reserved left for dB labels
const FREQ_MARGIN: f32 = 18.0; // px reserved bottom for freq labels

/// Font the corner readouts share, so their widths can be measured together.
fn readout_font() -> egui::FontId {
    egui::FontId::monospace(10.0)
}

/// Font of the analysis badge. A size larger than the readouts, because it is
/// transient and worth noticing.
fn badge_font() -> egui::FontId {
    egui::FontId::monospace(11.0)
}

/// Padding the analysis badge's background adds either side of its text.
const BADGE_PAD: f32 = 6.0;

/// The strip along the top of the plot that the corner readouts share.
///
/// LUFS, the channel legend and the frame rate were each anchored to a corner
/// by a different piece of code, and each assumed the row was empty. It was
/// not: at the default size "−23.4 LUFS" ran from the widget's left edge across
/// the `L`/`R` legend, which starts one dB-margin further in, and on a narrow
/// window the legend reached the frame rate as well.
///
/// They are laid out here instead — LUFS from the left, the frame rate from the
/// right, and the legend in whatever is left between them — and all three sit
/// *inside* the plot, so the dB margin stays free for the axis labels that a
/// Diff plot puts there.
#[derive(Clone, Copy, Debug)]
struct TopRow {
    /// Left edge of the LUFS readout.
    lufs_x: f32,
    /// Left edge of the space the legend may use.
    legend_x: f32,
    /// The legend must not reach this; beyond it is the frame rate.
    limit_x: f32,
    /// Right edge of the frame-rate readout.
    fps_x: f32,
    /// Right edge of the analysis badge's *text*, when one is being drawn. Its
    /// background extends [`BADGE_PAD`] further right.
    analysis_x: f32,
    y: f32,
}

impl TopRow {
    /// Measure the two fixed readouts and hand the legend what is left.
    ///
    /// Measured rather than assumed: "−23.4 LUFS" and "— LUFS" differ by a
    /// third, and a three-digit frame rate is wider than a two-digit one.
    fn plan(
        painter: &egui::Painter,
        plot_rect: Rect,
        lufs: Option<&str>,
        fps: &str,
        analysis: Option<&str>,
    ) -> Self {
        const PAD: f32 = 6.0;
        const GAP: f32 = 10.0;
        let width = |t: &str, font: egui::FontId| {
            painter
                .layout_no_wrap(t.to_string(), font, Color32::WHITE)
                .size()
                .x
        };
        let lufs_w = lufs.map(|t| width(t, readout_font())).unwrap_or(0.0);
        let fps_w = width(fps, readout_font());
        // The badge is the widest thing in the row and comes and goes, so it
        // takes the right end and everything else moves left of it rather than
        // the other way round — a frame rate that jumped sideways whenever an
        // analysis started would be worse than one that stays put.
        let badge_w = analysis
            .map(|t| width(t, badge_font()) + 2.0 * BADGE_PAD)
            .unwrap_or(0.0);

        let left = plot_rect.left() + PAD;
        let right = plot_rect.right() - PAD;
        let fps_x = if badge_w > 0.0 { right - badge_w - GAP } else { right };
        Self {
            lufs_x: left,
            legend_x: if lufs_w > 0.0 { left + lufs_w + GAP } else { left },
            limit_x: (fps_x - fps_w - GAP).max(left),
            fps_x,
            analysis_x: right - BADGE_PAD,
            y: plot_rect.top() + 4.0,
        }
    }
}

fn draw_db_labels(painter: &egui::Painter, plot_rect: Rect) {
    let db_levels = [0i32, -10, -20, -30, -40, -50, -60, -70];
    for db in db_levels {
        let v = (db as f32 + 80.0) / 80.0;
        let y = plot_rect.bottom() - v * plot_rect.height();
        painter.line_segment(
            [Pos2::new(plot_rect.left(), y), Pos2::new(plot_rect.right(), y)],
            Stroke::new(0.5, Color32::from_gray(40)),
        );
        painter.text(
            Pos2::new(plot_rect.left() - 4.0, y),
            egui::Align2::RIGHT_CENTER,
            format!("{db}"),
            egui::FontId::monospace(9.0),
            Color32::from_gray(80),
        );
    }
}

/// Frequency ticks along the bottom of an ordinary plot.
///
/// Placed through [`eligible_ticks`], which separates the two bounds: Nyquist
/// decides which ticks exist, the configured range decides where they go. See
/// that function for why conflating them moved every label off its bar.
fn draw_freq_labels(
    painter: &egui::Painter,
    plot_rect: Rect,
    sample_rate: u32,
    min_freq: f32,
    max_freq: f32,
    scale: freq_scale::FreqScale,
) {
    for (freq, t) in eligible_ticks(sample_rate, min_freq, max_freq, scale) {
        let x = plot_rect.left() + t * plot_rect.width();
        painter.line_segment(
            [Pos2::new(x, plot_rect.bottom()), Pos2::new(x, plot_rect.bottom() + 4.0)],
            Stroke::new(1.0, Color32::from_gray(60)),
        );
        text_inside(
            painter,
            Pos2::new(x, plot_rect.bottom() + 5.0),
            egui::Align2::CENTER_TOP,
            tick_label(freq),
            egui::FontId::monospace(9.0),
            Color32::from_gray(90),
        );
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Round `n` to the nearest power of two within [MIN_BAR_COUNT, MAX_BAR_COUNT].
fn snap_pow2(n: usize) -> usize {
    let clamped = n.clamp(MIN_BAR_COUNT, MAX_BAR_COUNT);
    // Find the closest power of two
    let lower = (1usize << clamped.ilog2()).max(MIN_BAR_COUNT);
    let upper = (lower * 2).min(MAX_BAR_COUNT);
    if clamped - lower < upper - clamped { lower } else { upper }
}

// ---------------------------------------------------------------------------
// Persisted spectrum view settings (~/.moosik/spectrum.json)
// ---------------------------------------------------------------------------

/// The subset of spectrum-window state worth remembering across launches — the
/// analysis/display knobs the user tweaks. Deliberately excludes `fft_size`
/// (auto-scaled per track in `on_play`) and anything runtime (textures, paths,
/// per-frame state). Every field has a `serde(default)` so an older or partial
/// file still loads.
#[derive(Serialize, Deserialize, Clone, PartialEq, Default)]
struct SpectrumSettings {
    #[serde(default)] mode:          SpectrumMode,
    #[serde(default)] style:         VizStyle,
    #[serde(default)] channel_view:  channels::ChannelView,
    #[serde(default)] loudness_mode: LoudnessMode,
    #[serde(default)] bar_count:     usize,
    #[serde(default)] bar_gap:       f32,
    #[serde(default)] window_fn:     WindowFn,
    #[serde(default)] smoothing:     f32,
    /// Pre-process retention. `Option`, deliberately.
    ///
    /// `None` means the file predates the setting, and the pre-process path
    /// ignored `smoothing` entirely back then — so there is no value to carry
    /// forward and inheriting 0.75 would hand the user a 174 ms response they
    /// never chose and never saw. A missing field takes the documented default
    /// instead. `#[serde(default)]` on a plain `f32` could not express that:
    /// it would produce 0.0, which is a real setting meaning Off.
    #[serde(default)] pre_smoothing: Option<f32>,
    /// Waterfall history depth, in seconds. `None` in files written before the
    /// control existed, which take the default rather than 0.
    #[serde(default)] waterfall_secs: Option<f32>,
    #[serde(default)] diff_layout:   channels::DiffLayout,
    #[serde(default)] min_freq:      f32,
    #[serde(default)] max_freq:      f32,
    #[serde(default)] interp_mode:   InterpolationMode,
    #[serde(default)] pad_factor:    usize,
    #[serde(default)] overlap:       f32,
    #[serde(default)] bar_mapping:   BarMappingMode,
    #[serde(default = "def_dsd_rate")] dsd_rate: u32,
    /// `None` here means Custom — the user is driving `aslt_cfg` by hand.
    #[serde(default)] aslt_preset:   Option<aslt::AsltPreset>,
    #[serde(default)] aslt_cfg:      aslt::AsltConfig,
    #[serde(default = "def_pre_fps")] pre_fps: f32,
    #[serde(default = "def_cache_budget")] cache_budget_gb: f32,
    #[serde(default)] show_fft:      bool,
    #[serde(default)] show_peak:     bool,
    #[serde(default)] show_art:      bool,
    #[serde(default)] show_cache:    bool,
    // Peak-hold config (colour stored as RGB so no egui serde feature is needed).
    // Defaults mirror PeakHoldConfig::default() so a settings file written
    // before these fields existed still restores sensible peak-hold values.
    #[serde(default = "def_true")]       peak_enabled:      bool,
    #[serde(default = "def_peak_hold")]  peak_hold_ms:      f32,
    #[serde(default = "def_peak_fall")]  peak_fall_speed:   f32,
    /// Whether the pre-process may use the GPU. `Auto` follows the per-machine
    /// calibration, which is the measured answer; the other two override it.
    #[serde(default)] gpu_mode: gpu_calib::GpuMode,
    /// Cores the pre-process may use. 0 = the default (`cores - 2`).
    #[serde(default)] worker_threads: usize,
    #[serde(default = "def_peak_accel")] peak_acceleration: f32,
    #[serde(default)]                    peak_decay_mode:   PeakDecayMode,
    #[serde(default = "def_peak_thick")] peak_thickness:    u8,
    #[serde(default = "def_peak_color")] peak_color:        [u8; 3],
}

fn def_true() -> bool { true }
fn def_dsd_rate() -> u32 { crate::dsd::decimate::DEFAULT_ANALYSIS_RATE }
fn def_pre_fps() -> f32 { DEFAULT_PRE_FPS }
fn def_cache_budget() -> f32 { DEFAULT_CACHE_BUDGET_GB }
fn def_peak_hold() -> f32 { 500.0 }
fn def_peak_fall() -> f32 { 3.0 }
fn def_peak_accel() -> f32 { 4.0 }
fn def_peak_thick() -> u8 { 2 }
fn def_peak_color() -> [u8; 3] { [255, 255, 255] }

// serde(default) needs Default impls for the enums used above.
impl Default for SpectrumMode      { fn default() -> Self { SpectrumMode::PreProcess } }
impl Default for VizStyle          { fn default() -> Self { VizStyle::Bars } }
impl Default for LoudnessMode      { fn default() -> Self { LoudnessMode::Flat } }
impl Default for WindowFn          { fn default() -> Self { WindowFn::Hann } }
impl Default for InterpolationMode { fn default() -> Self { InterpolationMode::None } }
impl Default for BarMappingMode    { fn default() -> Self { BarMappingMode::Cqt } }

/// Smallest power-of-two FFT size whose window spans at least 100 ms at
/// `sample_rate` (so analysis resolution stays consistent across rates).
fn auto_fft_size_for(sample_rate: u32) -> usize {
    let min_samples = (sample_rate / 10) as usize;
    [1024usize, 2048, 4096, 8192, 16384, 32768]
        .into_iter()
        .find(|&sz| sz >= min_samples)
        .unwrap_or(32768)
}

fn spectrum_settings_path() -> PathBuf {
    home_dir().join(".moosik").join("spectrum.json")
}

fn load_spectrum_settings() -> Option<SpectrumSettings> {
    std::fs::read_to_string(spectrum_settings_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

fn save_spectrum_settings(s: &SpectrumSettings) {
    let path = spectrum_settings_path();
    if let Some(p) = path.parent() { let _ = std::fs::create_dir_all(p); }
    if let Ok(json) = serde_json::to_string_pretty(s) {
        let _ = std::fs::write(path, json);
    }
}

// ---------------------------------------------------------------------------
// SpectrumWindow — the main public interface consumed by MoosikApp
// ---------------------------------------------------------------------------

pub struct SpectrumWindow {
    pub open: bool,
    pub mode: SpectrumMode,
    pub style: VizStyle,
    /// Which channel spectrum to draw. Held as what the user asked for, not as
    /// what is currently possible — an unavailable view falls back to Mix for
    /// the frame and returns by itself when the obstacle clears.
    pub channel_view: channels::ChannelView,
    /// Why the requested view is or is not being honoured, recomputed each
    /// accepted tick so the UI can explain itself without repeating the logic.
    channel_availability: channels::ChannelAvailability,
    /// Deinterleaved copies of the stereo tap, reused between frames.
    ch_left_pcm: Vec<f32>,
    ch_right_pcm: Vec<f32>,
    /// Which cached matrix the presentation state describes.
    ///
    /// A worker can finish while the player is paused, which installs a cache
    /// with nothing calling `tick_pre` to notice. Comparing revisions is how a
    /// handoff from the live-FFT fallback to a completed analysis becomes a
    /// discontinuity like any other, instead of the fallback's peaks and
    /// waterfall rows surviving into cached playback.
    seen_pre_revision: u64,
    /// Which mode the last tick ran in, so a switch can be recognised.
    ///
    /// The two modes read entirely different producers — a live FFT of the last
    /// window of audio, and a row of a matrix computed minutes ago — so a switch
    /// is a discontinuity in exactly the way a seek is, and nothing derived from
    /// the old one may survive into the new.
    last_mode: SpectrumMode,
    pub loudness_mode: LoudnessMode,
    pub analyzer: SpectrumAnalyzer,
    pub sample_buf: SampleBuf,
    pub bar_count: usize,
    pub bar_gap: f32,       // physical-pixel gap between bars (default 1)
    pub fft_size: usize,
    pub window_fn: WindowFn,
    pub smoothing: f32,
    /// Pre-process retention, persisted separately from `smoothing`.
    pub pre_smoothing: f32,
    /// How the Diff view is arranged: orientation, and which way round each of
    /// its two axes runs.
    pub diff_layout: channels::DiffLayout,
    /// How much of the rolling waterfall's history to keep, in seconds.
    ///
    /// Seconds rather than rows because rows per second differ between modes
    /// and settings — the same row count is two thirds of a second in one
    /// configuration and four seconds in another, which is not a thing anyone
    /// can choose meaningfully.
    pub waterfall_secs: f32,
    pub min_freq: f32,
    pub max_freq: f32,
    current_path: Option<PathBuf>,
    status_msg: String,
    pub max_fps: f32,
    last_fft_time: Option<Instant>,
    current_fps: f32,
    /// Instrumentation: wall-clock time of the previous show() call, the
    /// resulting real repaint rate (NOT the FFT throttle rate), and the
    /// measured cost of the last full draw. Surfaced in the F3 overlay so we
    /// can tell "too many frames" from "each frame too expensive".
    last_show_time: Option<Instant>,
    real_repaint_fps: f32,
    /// Cached rows consumed per second, smoothed for readability. Reported
    /// beside the tick rate because the two are no longer the same number:
    /// before this was fixed, a 60 Hz tick against a 180 fps cache displayed
    /// one row in three and the overlay had no way to show it.
    pre_rows_per_sec: f32,
    /// Wall clock the row counter was last sampled at.
    pre_rows_sampled: Option<Instant>,
    /// Channel transforms per second, so the cost of a Left/Right view is
    /// visible rather than inferred. Zero on Mix, by construction.
    channel_ffts_per_sec: f32,
    draw_ms: f32,
    pub waveform: Option<Vec<f32>>,
    waveform_rx: Option<std::sync::mpsc::Receiver<(Vec<f32>, TrackAnalysis)>>,
    pub spectral_ceiling: Option<SpectralCeiling>,
    /// True once compute_spectral_ceiling has run for the current pre_frames.
    /// Without this, a track whose ceiling comes back None (e.g. near-silent)
    /// would rescan every frame of the whole track on every UI tick.
    spectral_ceiling_attempted: bool,
    pub stereo_buf: StereoBuf,
    pub track_analysis: Option<TrackAnalysis>,
    pub momentary_lufs: f32,
    pub correlation: f32,
    spectrogram: Spectrogram,
    spectrogram_texture: Option<egui::TextureHandle>,
    // The waterfall texture is a ring: a new row overwrites the oldest in
    // place, and the two halves are drawn as two quads to put them back in
    // order. Rebuilding the whole image for one new row meant re-palettising
    // and re-uploading 1024x120 pixels at the frame rate to change 1024 of
    // them — about 99% of that work thrown away, every frame.
    waterfall_texture: Option<egui::TextureHandle>,
    /// Next ring row to overwrite.
    waterfall_head: usize,
    /// Bar count the texture was allocated for; a change forces a rebuild.
    waterfall_tex_w: usize,
    /// Height of the uploaded texture, so a change of history depth is noticed
    /// the same way a change of bar count is.
    waterfall_tex_h: usize,
    /// `Analyzer::waterfall_seq` as of the last upload.
    waterfall_uploaded_seq: u64,
    octave_bands: Vec<(f32, f32)>,
    octave_smoothed: Vec<f32>,
    auto_fft_size: usize,         // last FFT size chosen automatically; 0 = user overrode it
    phasescope_frames: Vec<[f32; 2]>, // snapshot captured in tick() — no lock/clone in show()
    /// Last momentary-LUFS computation — throttled by wall clock (~15 Hz) so
    /// the cost doesn't scale with the repaint rate.
    last_lufs_time: Option<Instant>,
    /// Reused buffer for the LUFS window snapshot (avoids a ~75 KB alloc per run).
    lufs_scratch: Vec<f32>,
    pub interp_mode: InterpolationMode,
    pub pad_factor: usize,
    pub overlap: f32,
    pub bar_mapping: BarMappingMode,
    pub show_debug: bool,
    /// GPU policy for the pre-process, mirrored into `gpu_calib`'s global.
    pub gpu_mode: gpu_calib::GpuMode,
    /// Cores the pre-process may use; 0 = default. Mirrored the same way.
    pub worker_threads: usize,
    /// Set while a re-calibration runs on a worker thread, so the button can
    /// disable itself instead of queueing probes.
    recalibrating: Option<std::sync::mpsc::Receiver<()>>,
    /// True when settings changed and no cache exists for the new combo.
    needs_reanalysis: bool,
    /// Cached (count, bytes) of all .spectrumcache files; refreshed lazily.
    cache_stats: (usize, u64),
    cache_stats_at: Option<Instant>,
    /// Full paths of every .spectrumcache file known to exist (same refresh cadence).
    cache_file_set: std::collections::HashSet<PathBuf>,

    // ── Parametric EQ ──────────────────────────────────────────────────────
    pub eq_state:         EqStateHandle,
    pub show_eq:          bool,
    pub eq_overlay:       EqOverlayMode,
    eq_dragging_node:     Option<usize>,
    eq_hovered_node:      Option<usize>,

    // ── EQ Presets ─────────────────────────────────────────────────────────
    pub preset_library:   EqPresetLibrary,
    /// Which preset is currently loaded (None = custom / unsaved state).
    pub active_preset:    Option<PresetRef>,
    /// True when the current bands differ from the loaded preset.
    pub preset_modified:  bool,
    /// Preset we are trying to switch to — triggers save-changes prompt.
    pending_preset_switch: Option<PresetRef>,
    /// Text buffer for naming a new preset.
    eq_save_name_buf:     String,
    /// Which preset is being renamed and the current edit buffer.
    eq_rename_state:      Option<(PresetRef, String)>,
    /// Which preset is awaiting delete confirmation.
    eq_confirm_delete:    Option<PresetRef>,
    /// Whether the "Save As New" row is expanded.
    eq_save_new_open:     bool,
    /// Scope selector for the "Save As New" row (Global vs Local).
    eq_save_new_scope:    PresetScope,

    // ── Bit-perfect mode ───────────────────────────────────────────────────
    /// When true, EQ is not applied to audio — reflect that in the spectrum display.
    pub bit_perfect: bool,

    // ── Appearance ─────────────────────────────────────────────────────────
    /// Colour ramp for the visualisers. Set by the app each frame from the
    /// persisted Appearance settings.
    pub palette_kind:   SpectrumPalette,
    /// Accent used by the `Accent` palette — the current track's album-art
    /// tint. Set by the app each frame.
    pub palette_accent: Color32,
    /// Open/closed state for the settings sections. They live in a single
    /// wrap-around toggle row (instead of stacked collapsibles) so the plot
    /// keeps as much height as possible; each section's body renders full-width
    /// below the row only when open.
    show_fft_settings:  bool,
    show_peak_settings: bool,
    show_art_settings:  bool,
    show_cache_settings: bool,
    /// Last snapshot written to spectrum.json, for change detection, plus a
    /// timestamp so a slider drag doesn't rewrite the file every frame.
    saved_settings:   SpectrumSettings,
    settings_save_at: Option<Instant>,

    // ── Album art ──────────────────────────────────────────────────────────
    pub art_settings: ArtSettingsStore,
    /// Texture of the currently-playing track's album art, if any.
    /// Set by MoosikApp before each call to show().
    pub current_art: Option<(egui::TextureId, u32, u32)>,

    // ── Peak hold ──────────────────────────────────────────────────────────
    pub peak_config:     PeakHoldConfig,
    /// Current peak level per bar (normalized 0–1).
    peak_vals:           Vec<f32>,
    /// Seconds the peak has been held at its current value (Linear/Gravity modes).
    peak_hold_timers:    Vec<f32>,
    /// Current fall velocity per bar (for Gravity mode).
    peak_velocities:     Vec<f32>,
    /// Current alpha per bar (for FadeOut mode, 0–1).
    peak_alphas:         Vec<f32>,
}

impl SpectrumWindow {
    /// Push whatever waterfall rows are new into the ring texture.
    ///
    /// Only the rows that actually arrived are palettised and uploaded. The
    /// previous version rebuilt the entire image — 1024×120 pixels — every time
    /// a single 1024-pixel row changed, at the frame rate.
    fn update_waterfall_texture(&mut self, ctx: &egui::Context, pal: &Palette) {
        let h = self.analyzer.waterfall_rows.max(1);
        let n = self.analyzer.waterfall.first().map(|r| r.len()).unwrap_or(0);
        if n == 0 {
            // Returning here left the previous texture in place and `draw_waterfall`
            // went on painting it — a rolling history of a track that had ended,
            // over a plot that had none. An empty ring means there is nothing to
            // show, which is a state the renderer has to be able to be in.
            self.discard_waterfall_texture();
            return;
        }

        // A bar-count change makes every stored row the wrong width, and a
        // depth change makes the image the wrong height; either way the texture
        // is thrown away and refilled from what the ring still holds.
        if self.waterfall_tex_w != n || self.waterfall_tex_h != h {
            self.waterfall_texture = None;
            self.waterfall_tex_w = n;
            self.waterfall_tex_h = h;
            self.waterfall_head = 0;
            self.waterfall_uploaded_seq =
                self.analyzer.waterfall_seq - self.analyzer.waterfall.len() as u64;
        }
        let th = self.waterfall_texture.get_or_insert_with(|| {
            ctx.load_texture(
                "moosik_waterfall",
                egui::ColorImage::new([n, h], Color32::BLACK),
                egui::TextureOptions::NEAREST,
            )
        });

        // Never more rows than the ring holds: after a seek or a stall the
        // counter can jump far ahead, and redrawing the ring twice over would
        // cost more than the whole optimisation saves.
        let new = (self.analyzer.waterfall_seq - self.waterfall_uploaded_seq)
            .min(h as u64).min(self.analyzer.waterfall.len() as u64) as usize;
        let first = self.analyzer.waterfall.len() - new;
        // Written *backwards* through the texture. The display puts the newest
        // row at the top and scrolls downward, so chronological order has to run
        // up the texture; writing forwards would silently invert the waterfall.
        for row in &self.analyzer.waterfall[first..] {
            th.set_partial([0, self.waterfall_head], waterfall_row_image(row, n, pal),
                           egui::TextureOptions::NEAREST);
            self.waterfall_head = (self.waterfall_head + h - 1) % h;
        }
        self.waterfall_uploaded_seq = self.analyzer.waterfall_seq;
        self.analyzer.waterfall_dirty = false;
    }

    /// Draw the ring as two quads, newest at the top.
    fn draw_waterfall(&self, painter: &egui::Painter, rect: Rect) {
        let Some(th) = &self.waterfall_texture else { return };
        let (split, top_uv, bot_uv) =
            waterfall_slices(self.waterfall_head, self.waterfall_tex_h.max(1));
        let uv = |(v0, v1): (f32, f32)| egui::Rect::from_min_max(
            egui::Pos2::new(0.0, v0), egui::Pos2::new(1.0, v1));
        let band = |y0: f32, y1: f32| Rect::from_min_max(
            egui::Pos2::new(rect.left(), y0), egui::Pos2::new(rect.right(), y1));

        let mid = rect.top() + rect.height() * split;
        if top_uv.1 > top_uv.0 {
            painter.image(th.id(), band(rect.top(), mid), uv(top_uv), Color32::WHITE);
        }
        if bot_uv.1 > bot_uv.0 {
            painter.image(th.id(), band(mid, rect.bottom()), uv(bot_uv), Color32::WHITE);
        }
    }

    pub fn new() -> Self {
        let buf = new_sample_buf();
        let analyzer = SpectrumAnalyzer::new(Arc::clone(&buf));
        let mut w = Self {
            open: false,
            mode: SpectrumMode::PreProcess,
            style: VizStyle::Bars,
            channel_view: channels::ChannelView::Mix,
            channel_availability: channels::ChannelAvailability::NoLiveTap,
            ch_left_pcm: Vec::new(),
            ch_right_pcm: Vec::new(),
            seen_pre_revision: 0,
            last_mode: SpectrumMode::PreProcess,
            loudness_mode: LoudnessMode::Flat,
            analyzer,
            sample_buf: buf,
            bar_count: DEFAULT_BAR_COUNT,
            bar_gap: 1.0,
            fft_size: DEFAULT_FFT_SIZE,
            window_fn: WindowFn::Hann,
            smoothing: 0.75,
            pre_smoothing: timing::DEFAULT_PRE_SMOOTHING,
            diff_layout: channels::DiffLayout::default(),
            waterfall_secs: timing::DEFAULT_WATERFALL_SECS,
            min_freq: DEFAULT_MIN_FREQ,
            max_freq: DEFAULT_MAX_FREQ,
            current_path: None,
            status_msg: String::new(),
            max_fps: 60.0,
            last_fft_time: None,
            current_fps: 0.0,
            last_show_time: None,
            real_repaint_fps: 0.0,
            pre_rows_per_sec: 0.0,
            pre_rows_sampled: None,
            channel_ffts_per_sec: 0.0,
            draw_ms: 0.0,
            waveform: None,
            waveform_rx: None,
            spectral_ceiling: None,
            spectral_ceiling_attempted: false,
            spectrogram: Spectrogram::new(),
            spectrogram_texture: None,
            waterfall_texture: None,
            waterfall_head: 0,
            waterfall_tex_w: 0,
            waterfall_tex_h: 0,
            waterfall_uploaded_seq: 0,
            octave_bands: Vec::new(),
            octave_smoothed: Vec::new(),
            stereo_buf: new_stereo_buf(),
            track_analysis: None,
            momentary_lufs: f32::NEG_INFINITY,
            correlation: 1.0,
            auto_fft_size: 0,
            phasescope_frames: Vec::new(),
            last_lufs_time: None,
            lufs_scratch: Vec::new(),
            interp_mode: InterpolationMode::None,
            pad_factor: 16,
            overlap: 0.875,
            bar_mapping: BarMappingMode::Cqt,
            show_debug: false,
            gpu_mode: gpu_calib::GpuMode::default(),
            worker_threads: 0,
            recalibrating: None,
            needs_reanalysis: false,
            cache_stats: (0, 0),
            cache_stats_at: None,
            cache_file_set: std::collections::HashSet::new(),
            eq_state: Arc::new(Mutex::new(EqState::new())),
            show_eq: false,
            eq_overlay: EqOverlayMode::Both,
            eq_dragging_node: None,
            eq_hovered_node: None,
            preset_library: EqPresetLibrary::load(),
            active_preset: None,
            preset_modified: false,
            pending_preset_switch: None,
            eq_save_name_buf: String::new(),
            eq_rename_state: None,
            eq_confirm_delete: None,
            eq_save_new_open: false,
            eq_save_new_scope: PresetScope::Global,
            bit_perfect: false,
            palette_kind: SpectrumPalette::Classic,
            palette_accent: Color32::from_rgb(0x94, 0xb1, 0xff),
            show_fft_settings: false,
            show_peak_settings: false,
            show_art_settings: false,
            show_cache_settings: false,
            saved_settings: SpectrumSettings::default(),
            settings_save_at: None,
            art_settings: ArtSettingsStore::load(),
            current_art: None,
            peak_config: PeakHoldConfig::default(),
            peak_vals: Vec::new(),
            peak_hold_timers: Vec::new(),
            peak_velocities: Vec::new(),
            peak_alphas: Vec::new(),
        };
        // Restore persisted view settings, then record the snapshot so we only
        // rewrite the file when something actually changes.
        if let Some(s) = load_spectrum_settings() {
            w.apply_settings(&s);
        }
        w.saved_settings = w.snapshot();
        w
    }

    /// Who computes the pre-process: how many cores, and whether the GPU helps.
    ///
    /// Both settings are visible and overridable rather than inferred silently,
    /// because both were constants baked in from one machine's measurements
    /// before this existed, and neither generalises: the GPU crossover is a
    /// property of a particular device against a particular core count, and the
    /// right number of worker threads depends on what else the machine is doing.
    fn compute_budget_ui(&mut self, ui: &mut egui::Ui) {
        let dark = ui.visuals().dark_mode;

        // Finished re-calibration hands back a token; until then the button is
        // disabled rather than able to queue a second probe behind the first.
        if let Some(rx) = &self.recalibrating
            && rx.try_recv().is_ok()
        {
            self.recalibrating = None;
        }
        let busy = self.recalibrating.is_some();

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Cores:");
            let cores = gpu_calib::core_count();
            let mut n = if self.worker_threads == 0 {
                gpu_calib::default_workers()
            } else {
                self.worker_threads
            };
            if ui.add(egui::Slider::new(&mut n, 1..=cores).suffix(format!(" / {cores}")))
                .on_hover_text(
                    "Cores the pre-process may use. The default leaves two free: \
                     the output thread has a hard deadline and the decoder feeds \
                     it, and a saturated machine is how an analysis turns into an \
                     audible dropout. The transform scales sub-linearly at the top \
                     end, so the last two cores buy less than they look like they \
                     would.")
                .changed()
            {
                self.worker_threads = n;
                gpu_calib::set_workers(n);
            }
            if ui.small_button("Default")
                .on_hover_text(format!("{} of {cores}", gpu_calib::default_workers()))
                .clicked()
            {
                self.worker_threads = 0;
                gpu_calib::set_workers(0);
            }
        });

        if !gpu_calib::device_present() {
            ui.label(egui::RichText::new("No GPU available — CPU route only.")
                .size(10.0).color(txt_dim(dark)));
            return;
        }

        ui.horizontal(|ui| {
            ui.label("GPU:");
            let mut m = self.gpu_mode;
            egui::ComboBox::from_id_salt("gpu_mode")
                .selected_text(match m {
                    gpu_calib::GpuMode::Auto   => "Auto (measured)",
                    gpu_calib::GpuMode::Always => "Always",
                    gpu_calib::GpuMode::Off    => "Off",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut m, gpu_calib::GpuMode::Auto, "Auto (measured)")
                        .on_hover_text(
                            "Use the device for exactly the block sizes it has been \
                             measured to win on this machine.");
                    ui.selectable_value(&mut m, gpu_calib::GpuMode::Always, "Always")
                        .on_hover_text(
                            "Send every eligible block size to the device. Not \
                             necessarily faster — on the smaller sizes the transfer \
                             costs more than the transform saves.");
                    ui.selectable_value(&mut m, gpu_calib::GpuMode::Off, "Off")
                        .on_hover_text("CPU only.");
                });
            if m != self.gpu_mode {
                self.gpu_mode = m;
                gpu_calib::set_mode(m);
            }

            if ui.add_enabled(!busy, egui::Button::new(if busy { "Measuring…" } else { "Re-measure" }))
                .on_hover_text(
                    "Throw away this machine's calibration and probe again. Takes \
                     a few seconds. Worth doing after a driver update, or if the \
                     machine was busy when it was first measured.")
                .clicked()
            {
                let (tx, rx) = std::sync::mpsc::channel();
                self.recalibrating = Some(rx);
                // Off the UI thread: the probe runs real transforms and holds
                // the calibration lock while it does.
                std::thread::Builder::new()
                    .name("moosik-gpu-probe".into())
                    .spawn(move || { gpu_calib::recalibrate(); let _ = tx.send(()); })
                    .ok();
            }
        });

        // What the machine has actually decided, per block size. "(probe)" means
        // the synthetic bootstrap is still in charge and real A/B samples are
        // still being collected.
        for line in gpu_calib::describe() {
            ui.label(egui::RichText::new(line).size(10.0).monospace().color(txt_dim(dark)));
        }
        let st = aslt::stats();
        if st.used_gpu() {
            ui.label(egui::RichText::new(format!(
                "last run: {:.1} s · device busy {:.0}% of its own phases · {} bars on device, {} on cores",
                st.total_s, st.device_duty() * 100.0, st.gpu_bars, st.cpu_bars,
            )).size(10.0).color(txt_dim(dark)));
        }
    }

    /// Current persistable view settings.
    fn snapshot(&self) -> SpectrumSettings {
        SpectrumSettings {
            mode:          self.mode.clone(),
            style:         self.style.clone(),
            channel_view:  self.channel_view,
            loudness_mode: self.loudness_mode.clone(),
            bar_count:     self.bar_count,
            bar_gap:       self.bar_gap,
            window_fn:     self.window_fn.clone(),
            smoothing:     self.smoothing,
            pre_smoothing: Some(self.pre_smoothing),
            waterfall_secs: Some(self.waterfall_secs),
            diff_layout:   self.diff_layout,
            min_freq:      self.min_freq,
            max_freq:      self.max_freq,
            interp_mode:   self.interp_mode.clone(),
            pad_factor:    self.pad_factor,
            overlap:       self.overlap,
            bar_mapping:   self.bar_mapping.clone(),
            dsd_rate:      self.analyzer.dsd_rate,
            aslt_preset:   self.analyzer.aslt_preset,
            aslt_cfg:      self.analyzer.aslt_cfg.clone(),
            pre_fps:       self.analyzer.pre_fps,
            cache_budget_gb: self.analyzer.cache_budget_gb,
            gpu_mode:       self.gpu_mode,
            worker_threads: self.worker_threads,
            show_fft:      self.show_fft_settings,
            show_peak:     self.show_peak_settings,
            show_art:      self.show_art_settings,
            show_cache:    self.show_cache_settings,
            peak_enabled:      self.peak_config.enabled,
            peak_hold_ms:      self.peak_config.hold_ms,
            peak_fall_speed:   self.peak_config.fall_speed,
            peak_acceleration: self.peak_config.acceleration,
            peak_decay_mode:   self.peak_config.decay_mode.clone(),
            peak_thickness:    self.peak_config.peak_thickness,
            peak_color: [
                self.peak_config.color.r(),
                self.peak_config.color.g(),
                self.peak_config.color.b(),
            ],
        }
    }

    /// Apply persisted view settings, mirroring the analyzer fields exactly as
    /// the UI handlers do, then rebuild so analysis reflects them.
    fn apply_settings(&mut self, s: &SpectrumSettings) {
        self.mode          = s.mode.clone();
        self.style         = s.style.clone();
        self.channel_view  = s.channel_view;
        self.loudness_mode = s.loudness_mode.clone();
        self.bar_gap       = s.bar_gap.clamp(0.0, 12.0);
        self.window_fn     = s.window_fn.clone();
        self.smoothing     = s.smoothing.clamp(0.0, 0.99);
        // A file without the field is not a file requesting 0.75.
        self.pre_smoothing = s.pre_smoothing
            .map(|v| v.clamp(0.0, 0.99))
            .unwrap_or(timing::DEFAULT_PRE_SMOOTHING);
        self.diff_layout = s.diff_layout;
        self.waterfall_secs = s.waterfall_secs
            .map(|v| v.clamp(*timing::WATERFALL_SECS_RANGE.start(),
                             *timing::WATERFALL_SECS_RANGE.end()))
            .unwrap_or(timing::DEFAULT_WATERFALL_SECS);
        self.min_freq      = s.min_freq;
        self.max_freq      = s.max_freq;
        self.interp_mode   = s.interp_mode.clone();
        self.pad_factor    = s.pad_factor.clamp(1, 64);
        self.overlap       = s.overlap;
        self.bar_mapping   = s.bar_mapping.clone();
        // A stored preset wins over a stored config: if the ladder's numbers are
        // ever retuned, saved settings should follow the new ladder rather than
        // pin users to the old one. Custom (`None`) keeps its explicit config.
        self.analyzer.aslt_preset = s.aslt_preset;
        self.analyzer.aslt_cfg = match s.aslt_preset {
            Some(p) => p.config(),
            None    => s.aslt_cfg.clone(),
        };
        self.analyzer.pre_fps = s.pre_fps.clamp(24.0, 480.0);
        self.analyzer.cache_budget_gb = s.cache_budget_gb.clamp(0.0, 200.0);
        // These two live in globals the transform reads, so the stored value has
        // to be pushed there as well as mirrored on the window.
        self.gpu_mode = s.gpu_mode;
        gpu_calib::set_mode(s.gpu_mode);
        self.worker_threads = s.worker_threads.min(gpu_calib::core_count());
        gpu_calib::set_workers(self.worker_threads);
        self.analyzer.dsd_rate = if crate::dsd::decimate::ANALYSIS_RATES.contains(&s.dsd_rate) {
            s.dsd_rate
        } else {
            crate::dsd::decimate::DEFAULT_ANALYSIS_RATE
        };
        self.show_fft_settings   = s.show_fft;
        self.show_peak_settings  = s.show_peak;
        self.show_art_settings   = s.show_art;
        self.show_cache_settings = s.show_cache;

        self.peak_config.enabled        = s.peak_enabled;
        self.peak_config.hold_ms        = s.peak_hold_ms.clamp(10.0, 1000.0);
        self.peak_config.fall_speed     = s.peak_fall_speed.clamp(0.05, 5.0);
        self.peak_config.acceleration   = s.peak_acceleration.clamp(0.5, 20.0);
        self.peak_config.decay_mode     = s.peak_decay_mode.clone();
        self.peak_config.peak_thickness = s.peak_thickness.clamp(1, 6);
        self.peak_config.color = Color32::from_rgb(s.peak_color[0], s.peak_color[1], s.peak_color[2]);

        // Mirror into the analyzer (fft_size is left alone — on_play auto-scales it).
        self.analyzer.window_fn   = self.window_fn.clone();
        self.analyzer.interp_mode = self.interp_mode.clone();
        self.analyzer.pad_factor  = self.pad_factor;
        self.analyzer.overlap     = self.overlap;
        self.analyzer.bar_mapping = self.bar_mapping.clone();
        let bc = s.bar_count.clamp(MIN_BAR_COUNT, MAX_BAR_COUNT);
        self.bar_count = bc;
        self.analyzer.set_bar_count(bc);
        self.sync_params();
        self.analyzer.rebuild_fft();
    }

    /// Save view settings when they change, throttled so a slider drag doesn't
    /// hammer the disk. Call once per frame at the end of `show`.
    fn persist_settings_if_changed(&mut self) {
        let cur = self.snapshot();
        if cur == self.saved_settings { return; }
        let ready = self.settings_save_at
            .map(|t| t.elapsed().as_secs_f32() > 0.6)
            .unwrap_or(true);
        if ready {
            save_spectrum_settings(&cur);
            self.saved_settings = cur;
            self.settings_save_at = Some(Instant::now());
        }
    }

    /// Recompute and push eq_weights (and other params) into the analyzer.
    fn sync_params(&mut self) {
        // Frequency range, mapping and loudness all change what a bar means, so
        // the channel filter state stops describing the same measurement.
        self.analyzer.reset_channels();
        self.analyzer.min_freq = self.min_freq;
        self.analyzer.max_freq = self.max_freq;
        self.analyzer.smoothing = self.smoothing;
        self.analyzer.pre_smoothing = self.pre_smoothing;
        self.analyzer.eq_weights = if self.loudness_mode == LoudnessMode::EqualLoudness {
            compute_eq_weights(self.bar_count, self.min_freq, self.max_freq,
                               self.analyzer.aslt_cfg.scale)
        } else {
            Vec::new()
        };
    }

    /// After any quality-setting change, check whether a cache file already exists
    /// for the current path + new settings.  If yes, load it instantly and return true.
    /// If no, clear pre_frames and set `needs_reanalysis` so the UI can prompt the user.
    /// Whether the FFT-only controls (size, window, padding, overlap,
    /// interpolation) affect the pre-processed result.
    ///
    /// They do not under Superlet, which analyses the time-domain signal
    /// directly. Their cache keys correctly ignore them — but that made every
    /// value of every one of them light up as "cached", since they all resolve
    /// to the same file. Technically true, thoroughly misleading, so the
    /// indicator is suppressed instead.
    fn fft_knobs_apply(&self) -> bool {
        self.bar_mapping != BarMappingMode::Superlet
    }

    fn try_load_or_flag_reanalysis(&mut self) {
        let Some(path) = self.current_path.clone() else { return; };
        let cache = cache_path_for(
            &path, self.bar_count, self.fft_size, self.pad_factor, self.overlap,
            &self.window_fn, self.min_freq, self.max_freq, &self.bar_mapping, &self.interp_mode,
            self.analyzer.dsd_rate, &self.analyzer.aslt_cfg, self.analyzer.pre_fps,
        );
        if let Some(frames) = load_cache(&cache, self.bar_count) {
            // Via pre_hop(), not fft_size: a superlet cache is spaced by frame
            // rate, and using the FFT hop here would drift the spectrum against
            // playback for the whole track.
            let rate = self.analyzer.sample_rate as f64 / self.analyzer.pre_hop() as f64;
            self.analyzer.set_pre_frames(frames, rate);
            self.spectral_ceiling = None; // recompute on next tick
            self.spectral_ceiling_attempted = false;
            self.needs_reanalysis = false;
        } else {
            self.analyzer.clear_pre_frames();
            if self.mode == SpectrumMode::PreProcess {
                self.needs_reanalysis = true;
            }
        }
    }

    // ── Preset helpers ──────────────────────────────────────────────────────

    fn track_key(&self) -> Option<String> {
        self.current_path.as_ref().map(|p| p.to_string_lossy().into_owned())
    }

    /// Load bands from a preset ref into EqState and mark it active.
    fn apply_preset_ref(&mut self, r: PresetRef) {
        let key = self.track_key();
        if let Some(preset) = self.preset_library.find(&r, key.as_deref()) {
            let bands = preset.bands.clone();
            let mut eq = self.eq_state.lock().unwrap();
            eq.bands = bands;
            eq.bump();
        }
        self.active_preset = Some(r);
        self.preset_modified = false;
    }

    /// Recompute `preset_modified` from the current EQ bands vs the active preset.
    fn refresh_preset_modified(&mut self) {
        let current = self.eq_state.lock().unwrap().bands.clone();
        self.preset_modified = match &self.active_preset {
            None => !current.is_empty(),
            Some(r) => {
                let key = self.track_key();
                match self.preset_library.find(r, key.as_deref()) {
                    Some(p) => !bands_equal(&current, &p.bands),
                    None    => true,
                }
            }
        };
    }

    /// Save current bands into the active preset (overwrite).
    fn overwrite_active_preset(&mut self) {
        let key = self.track_key();
        let bands = self.eq_state.lock().unwrap().bands.clone();
        if let Some(ref r) = self.active_preset.clone() {
            if let Some(p) = self.preset_library.find_mut(r, key.as_deref()) {
                p.bands = bands;
            }
            self.preset_library.save();
            self.preset_modified = false;
        }
    }

    /// Create a new preset from current bands, return its PresetRef.
    fn save_as_new_preset(&mut self, name: String, scope: PresetScope) -> PresetRef {
        let id = self.preset_library.alloc_id();
        let key = self.track_key();
        let bands = self.eq_state.lock().unwrap().bands.clone();
        let preset = EqPreset { id, name, bands };
        match scope {
            PresetScope::Global => self.preset_library.global.push(preset),
            PresetScope::Local => {
                let k = key.unwrap_or_default();
                self.preset_library.per_track.entry(k).or_default().push(preset);
            }
        }
        self.preset_library.save();
        let r = PresetRef { scope, id };
        self.active_preset = Some(r.clone());
        self.preset_modified = false;
        r
    }

    /// Call when loading a new track — tries to auto-load the right preset.
    fn auto_load_preset_for(&mut self, path: &Path) {
        let key = path.to_string_lossy().into_owned();
        let pref = self.preset_library.last_used.get(&key).cloned()
            .or_else(|| self.preset_library.default_id.map(|id| PresetRef { scope: PresetScope::Global, id }));
        match pref {
            Some(r) => {
                // Verify it still exists before applying
                let exists = self.preset_library.find(&r, Some(&key)).is_some();
                if exists { self.apply_preset_ref(r); }
                else {
                    // Preset was deleted — fall through to empty
                    let mut eq = self.eq_state.lock().unwrap();
                    eq.bands.clear(); eq.bump();
                    self.active_preset = None; self.preset_modified = false;
                }
            }
            None => {
                let mut eq = self.eq_state.lock().unwrap();
                eq.bands.clear(); eq.bump();
                self.active_preset = None; self.preset_modified = false;
            }
        }
    }

    /// Call just before switching away from the current track.
    fn persist_last_used(&mut self) {
        if let Some(key) = self.track_key()
            && let Some(ref r) = self.active_preset {
            self.preset_library.last_used.insert(key, r.clone());
            self.preset_library.save();
        }
    }

    /// Call when a new track starts. `sample_rate` is from the rodio Decoder.
    pub fn on_play(&mut self, path: &Path, sample_rate: u32) {
        // Save the active preset for the track we're leaving.
        self.persist_last_used();

        // DSD plays via DoP (the engine reports the carrier rate), but the
        // analyzer works on the decimated PCM feed — its rate is what every
        // downstream consumer (auto-FFT, frame timing, axis labels) must see.
        let sample_rate = if crate::dsd::is_dsd_path(path) {
            crate::dsd::parse_file(path)
                .map(|i| crate::dsd::decimate::analysis_rate_for(i.sample_rate, self.analyzer.dsd_rate))
                .unwrap_or(sample_rate)
        } else {
            sample_rate
        };

        // Auto-scale FFT size so the analysis window is 100–200 ms at the
        // track's sample rate (e.g. 8192 @ 44.1/48 kHz ≈ 170–185 ms,
        // 16384 @ 96 kHz ≈ 170 ms, 32768 @ 192 kHz ≈ 170 ms).
        let auto_fft = auto_fft_size_for(sample_rate);
        self.fft_size = auto_fft;
        self.analyzer.fft_size = auto_fft;
        self.analyzer.rebuild_fft();
        self.auto_fft_size = auto_fft;

        self.current_path = Some(path.to_path_buf());
        // Auto-load the last-used (or default) preset for the new track.
        self.auto_load_preset_for(path);
        // A repeat, or a restart from the top, calls this for the file already
        // being analysed. Cancelling and relaunching there means a superlet run
        // can never outlive the track it is analysing — on loop it restarts for
        // ever and never finishes. Recognise that case and let it run.
        let resuming = self.analyzer.is_analyzing.load(Ordering::Relaxed)
            && self.analyzer.analyzing_path.as_deref() == Some(path);
        if resuming {
            self.analyzer.reset_keeping_analysis();
        } else {
            self.analyzer.reset();
        }
        self.analyzer.sample_rate = sample_rate;
        self.analyzer.bar_count = self.bar_count;
        self.sync_params();
        self.spectral_ceiling = None;
        self.spectral_ceiling_attempted = false;
        self.waveform = None;
        self.waveform_rx = None;
        self.track_analysis = None;
        self.needs_reanalysis = false;
        // The window's half of the presentation state, which `analyzer.reset()`
        // above cannot reach: peak decay and the uploaded waterfall texture.
        // Without this a peak marker from the previous track hung over the new
        // one and the GPU ring kept drawing rows from a track that had ended.
        self.reset_presentation();
        // The stereo lease is deliberately *not* touched here.
        //
        // This is UI metadata, and it runs after the route has already been
        // built: the engine restarts playback, which constructs the tap, and
        // only then does the app call this with the new track's title and rate.
        // Retiring the lease here bumped the generation past the tap that had
        // just been created, so every write it made was rejected for the whole
        // track and Left/Right never worked on ordinary playback at all.
        //
        // Ownership belongs to the route, which is the only thing that knows
        // what is actually feeding the analyser: a tap claims when it is first
        // pulled and gives the lease back when it is dropped, the engine
        // retires it after tearing every route down, and the native DSD paths
        // publish `NoLiveTap` because they attach no tap at all.
        // Always preprocess — needed for spectral ceiling even in real-time mode.
        // Skipped while resuming: the existing thread is already doing exactly
        // this work, and start_preprocess would only be turned away by its own
        // is_analyzing guard anyway.
        if !resuming {
            self.analyzer.start_preprocess(path.to_path_buf());
        }
        self.waveform_rx = None;
    }

    /// Call when playback stops or the player is stopped.
    ///
    /// The lease is the engine's to retire — it is the thing that tore the
    /// route down — but the derived channel spectra are this window's, and
    /// leaving them on screen would show a stopped player a live-looking
    /// left and right.
    pub fn on_stop(&mut self) {
        self.analyzer.reset();
        self.reset_presentation();
        self.channel_availability = channels::ChannelAvailability::NoLiveTap;
    }

    /// Call every UI frame to advance FFT / pre-process state.
    ///
    /// Internally throttled to `max_fps`; the FPS counter reflects the actual
    /// FFT run rate, not the UI repaint rate (which can spike when the mouse
    /// moves over the spectrum viewport).
    pub fn tick(&mut self, elapsed_secs: f64, is_playing: bool) {
        // Poll background results — always, even when paused
        if let Some(ref rx) = self.waveform_rx
            && let Ok((wf, analysis)) = rx.try_recv() {
            self.waveform = Some(wf);
            self.track_analysis = Some(analysis);
            self.waveform_rx = None;
        }
        self.analyzer.try_receive_frames();
        // Drain waveform + analysis that arrived via PreMessage::Done
        if let Some(wf) = self.analyzer.pending_waveform.take() {
            self.waveform = Some(wf);
        }
        if let Some(analysis) = self.analyzer.pending_analysis.take() {
            self.track_analysis = Some(analysis);
        }
        if !self.spectral_ceiling_attempted && !self.analyzer.pre_frames.is_empty() {
            self.spectral_ceiling = compute_spectral_ceiling(
                &self.analyzer.pre_frames, self.bar_count,
                self.min_freq, self.max_freq.min(self.analyzer.sample_rate as f32 / 2.0),
                self.analyzer.aslt_cfg.scale,
            );
            self.spectral_ceiling_attempted = true;
        }

        // Before the guard below, deliberately. A stopped or paused player still
        // has to tell the truth about what it could show: stopping retires the
        // stereo lease, so leaving Left/Right enabled and frozen on the last
        // frame of a track that is no longer playing is the same stale claim as
        // a "Playing:" line on a stopped player. Changing mode or visualisation
        // while paused has to take effect then, too, not on the next play.
        self.refresh_channel_state();
        self.refresh_waterfall_depth();
        // A cache can be installed by a worker at any moment, including while
        // paused. Checked here rather than where the frames arrive because
        // there are three places they arrive from and one place that owns the
        // state they invalidate.
        if self.seen_pre_revision != self.analyzer.pre_revision() {
            self.reset_presentation();
        }

        if !is_playing { return; }

        // Throttle FFT runs to max_fps
        let min_dt = 1.0 / self.max_fps.max(1.0) as f64;
        let now = Instant::now();
        let elapsed_since_last = self.last_fft_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(f64::MAX);
        if elapsed_since_last < min_dt {
            return; // too soon — skip this repaint
        }

        // Measure the actual analyser tick rate (not the UI repaint rate)
        if elapsed_since_last < 5.0 {
            let dt = elapsed_since_last as f32;
            self.current_fps = self.current_fps * 0.85 + (1.0 / dt) * 0.15;
        }
        self.last_fft_time = Some(now);

        // Wall time this tick represents. Smoothing is defined per unit of
        // elapsed time, so this is what the real-time path filters with; the
        // pre-process path ignores it and uses source time instead, and only
        // passes it along for its live-FFT fallback. The first tick of a stream
        // arrives with `f64::MAX`, which resolves to no smoothing — correct,
        // since there is no previous frame to retain.
        let tick_dt = elapsed_since_last;

        // Rows-per-second, sampled on the analyser clock rather than derived
        // from the tick rate, so it stays honest when the two diverge.
        let rows = self.analyzer.take_rows_consumed();
        let ch_ffts = self.analyzer.take_channel_ffts();
        if let Some(at) = self.pre_rows_sampled {
            let span = at.elapsed().as_secs_f32();
            if span > 0.25 {
                self.pre_rows_per_sec = rows as f32 / span;
                self.channel_ffts_per_sec = ch_ffts as f32 / span;
                self.pre_rows_sampled = Some(now);
            } else {
                // Not enough time to divide by; give the counts back so the
                // next sample includes them rather than losing them.
                self.analyzer.add_rows_consumed(rows);
                self.analyzer.add_channel_ffts(ch_ffts);
            }
        } else {
            self.pre_rows_sampled = Some(now);
        }

        // Only maintain the waterfall ring-buffer when the waterfall view is actually displayed.
        // Skipping it when not needed saves ~4KB/tick of allocation+shift work.
        self.analyzer.waterfall_enabled = self.style == VizStyle::Waterfall;

        match self.mode {
            SpectrumMode::RealTime => {
                self.analyzer.process_realtime(tick_dt);
                self.tick_channels(tick_dt);
                // Feed spectrogram and octave bands from fresh FFT norms.
                // Real-time uses 2× padding (not pad_factor) — match here.
                if !self.analyzer.last_fft_norms.is_empty() {
                    let eff_fft = self.analyzer.fft_size * 2;
                    let pal = Palette::new(self.palette_kind, self.palette_accent);
                    self.spectrogram.push_frame(
                        &self.analyzer.last_fft_norms,
                        self.analyzer.sample_rate,
                        eff_fft,
                        self.min_freq, self.max_freq,
                        &pal,
                    );
                    let raw = octave_band_magnitudes(
                        &self.analyzer.last_fft_norms,
                        self.analyzer.sample_rate,
                        eff_fft,
                    );
                    if self.octave_smoothed.len() != raw.len() {
                        self.octave_smoothed = vec![0.0; raw.len()];
                        self.octave_bands = raw.iter().map(|&(f, _)| (f, 0.0)).collect();
                    }
                    // The decision this frame was folded in with, not a
                    // second one derived here. The meters used to smooth
                    // independently, so after a reset they crept up from zero
                    // behind bars that had already snapped.
                    let alpha = self
                        .analyzer
                        .frame_alpha()
                        .unwrap_or_else(|| self.smoothing.clamp(0.0, 1.0));
                    let _ = tick_dt;
                    for (i, &(fc, v)) in raw.iter().enumerate() {
                        self.octave_smoothed[i] = self.octave_smoothed[i] * alpha + v * (1.0 - alpha);
                        self.octave_bands[i] = (fc, self.octave_smoothed[i]);
                    }
                }
            }
            SpectrumMode::PreProcess => {
                // The v3 cache holds one mono row per frame, so there is nothing
                // to separate; `refresh_channel_state` has already said so.
                self.analyzer.reset_channels();
                self.analyzer.tick_pre(elapsed_secs, tick_dt);
            }
        }

        // Momentary LUFS (400 ms window from live sample buffer).
        // Throttled by wall clock to ~15 Hz — plenty for a meter display, and
        // independent of the repaint rate (was every other tick, which at high
        // max-fps settings ran the K-weight filter over 19k samples 100+ times
        // a second). LUFS is also used by the main-window readout so we keep
        // it running even when the spectrum panel is not in phasescope mode.
        let lufs_due = self.last_lufs_time.map(|t| t.elapsed().as_millis() >= 66).unwrap_or(true);
        if lufs_due {
            self.last_lufs_time = Some(Instant::now());
            let window = (self.analyzer.sample_rate as f64 * 0.4) as usize;
            self.lufs_scratch.clear();
            {
                let guard = self.sample_buf.lock().unwrap_or_else(|p| p.into_inner());
                if guard.len() >= window {
                    let start = guard.len() - window;
                    self.lufs_scratch.extend_from_slice(&guard[start..]);
                }
            } // lock released here
            if !self.lufs_scratch.is_empty() {
                let mut kw = KWeightFilter::new(self.analyzer.sample_rate);
                let mean_sq: f64 = self.lufs_scratch.iter()
                    .map(|&s| { let y = kw.process(s) as f64; y * y })
                    .sum::<f64>() / self.lufs_scratch.len() as f64;
                self.momentary_lufs = if mean_sq > 1e-10 {
                    (-0.691 + 10.0 * mean_sq.log10()) as f32
                } else {
                    f32::NEG_INFINITY
                };
            }
        }
        // Stereo correlation + phasescope snapshot.
        // Only needed when the phasescope view is active — skip the 32 KB lock+memcpy
        // and the 4096-iteration correlation computation in all other modes.
        if self.style == VizStyle::Phasescope {
            {
                let guard = self.stereo_buf.lock().unwrap_or_else(|p| p.into_inner());
                let guard = &guard.frames;
                let n = guard.len().min(4096);
                let start = guard.len().saturating_sub(n);
                self.phasescope_frames.clear();
                self.phasescope_frames.extend_from_slice(&guard[start..]);
            } // lock released here
            let n = self.phasescope_frames.len();
            if n >= 2 {
                let (mut lr, mut ll, mut rr) = (0.0f64, 0.0f64, 0.0f64);
                for &[l, r] in &self.phasescope_frames {
                    lr += (l * r) as f64; ll += (l * l) as f64; rr += (r * r) as f64;
                }
                let denom = (ll * rr).sqrt();
                self.correlation = if denom > 1e-10 { (lr / denom).clamp(-1.0, 1.0) as f32 } else { 1.0 };
            }
        }

        // Advance peak hold state; cap dt to avoid huge jumps after hiccups
        let peak_dt = elapsed_since_last.min(0.1) as f32;
        self.update_peaks(peak_dt);
    }

    /// Call after the user seeks to a new position. Flushes stale audio state
    /// and snaps the pre-process display to the correct frame.
    pub fn on_seek(&mut self, elapsed_secs: f64) {
        // Flush the realtime sample ring buffer — it still contains audio from
        // before the seek point and would show the wrong spectrum until refilled.
        match self.sample_buf.lock() {
            Ok(mut buf) => buf.clear(),
            Err(p) => p.into_inner().clear(),
        }
        // Everything on screen describes the position we just left — including
        // the frame cursor, so the next tick cannot try to cascade from wherever
        // the filter had reached to wherever the user landed, which for a
        // backward seek is not even forwards.
        self.reset_presentation();
        // In pre-process mode, immediately snap the display to the new position
        if self.mode == SpectrumMode::PreProcess {
            self.analyzer.snap_pre_to(elapsed_secs);
        }
    }

    /// Draw one series in `rect` using the current style.
    ///
    /// Kept separate from the Mix path because that one also carries album-art
    /// masking and peak hold, neither of which has a per-channel form yet.
    fn draw_one_series(
        &self,
        painter: &egui::Painter,
        mags: &[f32],
        rect: Rect,
        pal: &Palette,
        tint: Option<Color32>,
    ) {
        match (self.style.clone(), tint) {
            // An overlay of filled areas or bars hides whichever channel is
            // drawn second. Two lines is the only form that shows both, so an
            // overlay is always lines whatever the style selector says.
            (_, Some(c)) => draw_line(painter, mags, rect, c),
            (VizStyle::Bars, None) => draw_bars(painter, mags, rect, self.bar_gap, pal),
            (VizStyle::FilledArea, None) => draw_filled(painter, mags, rect, pal),
            (_, None) => draw_line(painter, mags, rect, pal.line()),
        }
    }

    /// Channel legend, so a single-channel or stacked plot cannot be mistaken
    /// for the Mix.
    ///
    /// Takes the position outright. It used to derive one from a rectangle,
    /// which is how it ended up sharing a corner with the LUFS readout: neither
    /// caller could see the other, and both were right about their own
    /// rectangle. Returns the width used, so a second legend can follow the
    /// first.
    fn label_channel(
        &self,
        painter: &egui::Painter,
        at: Pos2,
        limit_x: f32,
        text: &str,
        col: Color32,
    ) -> f32 {
        // Past the limit the frame rate begins; dropping the legend is better
        // than printing it over another number.
        if at.x >= limit_x {
            return 0.0;
        }
        let r = painter.text(
            at,
            egui::Align2::LEFT_TOP,
            text,
            egui::FontId::proportional(12.0),
            col,
        );
        r.width()
    }

    /// Render `view`, returning false if it could not be drawn and the caller
    /// should fall back to the Mix.
    ///
    /// Every branch takes its data from `frame`, which carries `None` rather
    /// than a copy of the mix when a channel does not exist — so there is no
    /// path here that can draw one series twice and label it L and R.
    fn draw_channel_view(
        &self,
        painter: &egui::Painter,
        view: channels::ChannelView,
        frame: &ChannelFrame<'_>,
        plot_rect: Rect,
        pal: &Palette,
        top_row: TopRow,
    ) -> bool {
        use channels::ChannelView as CV;
        // One guard for every view: whatever it draws must actually be present.
        // Falling back to the Mix is the only alternative — substituting the
        // mix for a missing channel is exactly the lie this whole path exists
        // to prevent.
        if (view.draws_left() && frame.left.is_none())
            || (view.draws_right() && frame.right.is_none())
        {
            return false;
        }
        match view {
            CV::Mix => false,
            CV::Left | CV::Right => {
                let (mags, name, col) = if view == CV::Left {
                    (frame.left, "L", CHANNEL_L_COLOR)
                } else {
                    (frame.right, "R", CHANNEL_R_COLOR)
                };
                let Some(mags) = mags else { return false };
                self.draw_one_series(painter, mags, plot_rect, pal, None);
                self.label_channel(painter, Pos2::new(top_row.legend_x, top_row.y), top_row.limit_x, name, col);
                true
            }
            CV::Split => {
                let Some((l, r)) = frame.pair() else { return false };
                // One divider, two equal halves: identical height means
                // identical dB scale, and the shared plot_rect width means one
                // frequency axis. Neither can drift from the other because
                // neither is computed twice.
                let mid = plot_rect.center().y;
                let upper = Rect::from_min_max(plot_rect.min, Pos2::new(plot_rect.right(), mid - 1.0));
                let bot = Rect::from_min_max(Pos2::new(plot_rect.left(), mid + 1.0), plot_rect.max);
                self.draw_one_series(painter, l, upper, pal, None);
                self.draw_one_series(painter, r, bot, pal, None);
                painter.line_segment(
                    [Pos2::new(plot_rect.left(), mid), Pos2::new(plot_rect.right(), mid)],
                    Stroke::new(1.0, Color32::from_gray(48)),
                );
                // Only the upper legend is in the shared row; the lower one
                // sits under the divider, where nothing else draws.
                self.label_channel(
                    painter,
                    Pos2::new(top_row.legend_x, top_row.y),
                    top_row.limit_x,
                    "L",
                    CHANNEL_L_COLOR,
                );
                self.label_channel(
                    painter,
                    Pos2::new(plot_rect.left() + 6.0, bot.top() + 4.0),
                    plot_rect.right(),
                    "R",
                    CHANNEL_R_COLOR,
                );
                true
            }
            CV::Diff => {
                let Some((l, r)) = frame.pair() else {
                    return false;
                };
                draw_channel_diff(painter, l, r, plot_rect, self.bar_gap, self.diff_layout);
                true
            }
            CV::Overlay => {
                let Some((l, r)) = frame.pair() else { return false };
                self.draw_one_series(painter, l, plot_rect, pal, Some(CHANNEL_L_COLOR));
                self.draw_one_series(painter, r, plot_rect, pal, Some(CHANNEL_R_COLOR));
                // Legend, because two unlabelled curves are a puzzle. The
                // second follows the width of the first rather than a guessed
                // offset, so a wider glyph cannot push them into each other.
                let w = self.label_channel(
                    painter,
                    Pos2::new(top_row.legend_x, top_row.y),
                    top_row.limit_x,
                    "L",
                    CHANNEL_L_COLOR,
                );
                self.label_channel(
                    painter,
                    Pos2::new(top_row.legend_x + w + 6.0, top_row.y),
                    top_row.limit_x,
                    "R",
                    CHANNEL_R_COLOR,
                );
                true
            }
        }
    }

    /// Whether the current visualisation has a per-channel form.
    ///
    /// The rolling histories — waterfall, spectrogram — and the meters keep one
    /// series each in this phase, and the phasescope is already a stereo
    /// instrument that Channel View has no business redefining. Those show the
    /// Mix and say so rather than going blank.
    fn style_supports_channels(&self) -> bool {
        matches!(
            self.style,
            VizStyle::Bars | VizStyle::Line | VizStyle::FilledArea
        )
    }

    /// Everything on screen stops describing the thing it was describing.
    ///
    /// The presentation state was split in two and only half of it was ever
    /// reset. The analyser owns the spectra, the smoothers and the cached-frame
    /// cursor; the *window* owns peak decay — values, hold timers, velocities
    /// and fade alphas — and the uploaded waterfall texture with its head and
    /// its upload watermark. `on_play` and `on_stop` reset the analyser and
    /// left the window's half alive, so a peak marker from the previous track
    /// hung over the new one and the GPU ring went on drawing rows from a
    /// track that had ended.
    ///
    /// One operation for every producer change: a new track, a stop, a seek, a
    /// mode switch, and a cache arriving or being taken away.
    ///
    /// It deliberately does not touch `pre_frames`. A cache that has just been
    /// installed is the *reason* for the discontinuity, not a casualty of it.
    fn reset_presentation(&mut self) {
        let n = self.analyzer.bar_count.max(1);
        // ── the bars, and everything derived from the last transform ──────
        self.analyzer.magnitudes = vec![0.0; n];
        self.analyzer.smoothed = vec![0.0; n];
        self.analyzer.peak_input = vec![0.0; n];
        // The raw half-spectrum the spectrogram and the octave meters are fed
        // from. Left in place it seeded both of them with the previous
        // producer's frame on the first tick after the reset.
        self.analyzer.last_fft_norms.clear();
        self.analyzer.reset_channels();
        // Cursor and the row the display was showing, together.
        self.analyzer.invalidate_pre_cursor();
        self.analyzer.snap_next_realtime_frame();

        // ── rolling histories, CPU side and GPU side ──────────────────────
        self.analyzer.waterfall.clear();
        self.analyzer.waterfall_dirty = false;
        self.discard_waterfall_texture();
        // The spectrogram is a second rolling history with a second texture,
        // and it was not reset at all: after a stop it went on painting a
        // ring of the track that had ended, and after a track change the new
        // one scrolled in beside the old.
        self.spectrogram.clear();
        self.spectrogram_texture = None;

        // ── the meters ────────────────────────────────────────────────────
        self.octave_bands.clear();
        self.octave_smoothed.clear();
        self.phasescope_frames.clear();
        self.correlation = 1.0;
        self.momentary_lufs = f32::NEG_INFINITY;
        self.lufs_scratch.clear();

        self.reset_peaks();
        // Whatever revision is installed now is the one this state describes.
        self.seen_pre_revision = self.analyzer.pre_revision();
    }

    /// Throw the uploaded waterfall away, and forget everything about it.
    ///
    /// The watermark has to move with the texture. Leaving it behind means the
    /// next upload believes the rows it already sent are still on the GPU, and
    /// uploads only the difference into an image that no longer exists.
    fn discard_waterfall_texture(&mut self) {
        self.waterfall_texture = None;
        self.waterfall_tex_w = 0;
        self.waterfall_tex_h = 0;
        self.waterfall_head = 0;
        self.waterfall_uploaded_seq = self.analyzer.waterfall_seq;
    }

    /// How many updates a second the thing feeding the waterfall produces.
    ///
    /// Rows per second the ring is sized against, and whether that number is
    /// measured or merely requested.
    ///
    /// The two modes differ in kind, not just in value:
    ///
    /// * **Pre-process** returns the cache's own frame rate. Every cached row is
    ///   consumed, so this is a measurement and the span it implies is the span
    ///   on screen.
    /// * **Real-time** returns `max_fps`, which is a **ceiling the analyser is
    ///   asked to respect, not a rate anything has achieved**. Nothing here
    ///   measures the accepted tick rate. If the machine cannot keep up — a busy
    ///   pre-process, a heavy superlet, a stall — fewer rows arrive per second,
    ///   and because the ring is a fixed number of rows the retained span gets
    ///   *longer* than the slider says, not shorter.
    ///
    /// Deliberately not fixed by measuring: a measured rate would resize the
    /// ring as the machine breathed, which is a resampler and an adaptive
    /// resize, and both were ruled out. The honest fix is to say which number
    /// this is, which the second return value carries to the UI.
    fn waterfall_row_rate(&self) -> (f64, bool) {
        match self.mode {
            SpectrumMode::PreProcess if !self.analyzer.pre_frames.is_empty() => {
                (self.analyzer.pre_frame_rate, false)
            }
            _ => (self.max_fps.max(1.0) as f64, true),
        }
    }

    /// Resize the ring when the wanted span or the producer rate changes.
    ///
    /// Cheap: a multiply and a comparison. It has to run every tick because the
    /// rate moves under it — a mode switch, a new cache at a different
    /// `pre_fps`, or the Max FPS slider all change how many rows a second
    /// arrive, and the span is the thing being held constant.
    fn refresh_waterfall_depth(&mut self) {
        let rows =
            timing::waterfall_rows_for(self.waterfall_secs, self.waterfall_row_rate().0);
        if rows == self.analyzer.waterfall_rows {
            return;
        }
        self.analyzer.waterfall_rows = rows;
        // Trim immediately so the ring never sits above its cap, and drop the
        // texture: its height is wrong now, and the upload path diffs against a
        // watermark that assumes the image it already sent is still valid.
        while self.analyzer.waterfall.len() > rows.max(1) {
            self.analyzer.waterfall.remove(0);
        }
        self.discard_waterfall_texture();
    }

    /// Bring the channel state up to date, whatever the transport is doing.
    ///
    /// Cheap enough to run on every tick: one lock on the tap to read a channel
    /// count, and a comparison. It performs no transforms — producing the
    /// spectra is `tick_channels`, which only runs while something is playing.
    fn refresh_channel_state(&mut self) {
        if self.mode != self.last_mode {
            self.on_mode_changed();
        }
        let tap_channels = self.live_tap_channels();
        let avail = channels::availability(
            self.mode == SpectrumMode::PreProcess,
            self.style_supports_channels(),
            tap_channels,
        );
        self.set_channel_availability(avail);
    }

    /// Switching between Real-time and Pre-process is a discontinuity.
    ///
    /// The two read unrelated producers, so nothing derived from one describes
    /// the other. Leaving the pre-process cursor in place meant switching away
    /// and back replayed however many cached rows the clock had crossed in the
    /// meantime — up to the catch-up limit — as a burst of history nobody
    /// played, and the waterfall kept rows from a scale that no longer applied.
    fn on_mode_changed(&mut self) {
        self.last_mode = self.mode.clone();
        // The same discontinuity every other producer change gets: the cursor
        // goes, so the first tick in the new mode snaps rather than cascading
        // from wherever the other producer had reached, and the first computed
        // real-time frame is shown unsmoothed rather than fading up from zero.
        self.reset_presentation();
    }

    /// What the live tap says it is carrying, or `NO_LIVE_TAP` if nothing is.
    fn live_tap_channels(&self) -> u16 {
        match self.stereo_buf.lock() {
            Ok(g) => g.channels,
            Err(p) => p.into_inner().channels,
        }
    }

    /// Adopt a new availability, resetting the channel spectra if it changed.
    ///
    /// Losing availability has to clear them: the alternative is a frozen left
    /// and right from a track, a mode or a device that is no longer current.
    fn set_channel_availability(&mut self, avail: channels::ChannelAvailability) {
        if avail != self.channel_availability {
            self.analyzer.reset_channels();
            self.channel_availability = avail;
        }
    }

    /// Decide whether L/R can be shown this tick, and produce them if so.
    ///
    /// The channel count comes from the tap, which took it from the decoder or
    /// the device format. Nothing here infers stereo from the buffer holding
    /// data: after a track change the buffer is emptied and marked as having no
    /// live tap, so a native-DSD track that attaches no tap at all reports
    /// honestly instead of redrawing the previous track.
    fn tick_channels(&mut self, dt: f64) {
        let want = self.channel_view.needs_channels();
        let fft_size = self.analyzer.fft_size;

        // Availability was decided by `refresh_channel_state`, which runs every
        // tick whether or not anything is playing. Re-deciding it here would be
        // the second copy of a rule that already has one home.
        let tap_channels = {
            let guard = match self.stereo_buf.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            let ch = guard.channels;
            // Copy under the same lock that read the channel count, so the two
            // cannot describe different streams.
            if want && ch == 2 {
                channels::deinterleave(
                    &guard.frames, fft_size,
                    &mut self.ch_left_pcm, &mut self.ch_right_pcm,
                );
            }
            ch
        };

        let avail = channels::availability(
            self.mode == SpectrumMode::PreProcess,
            self.style_supports_channels(),
            tap_channels,
        );
        self.set_channel_availability(avail);

        if want && self.channel_availability.is_available() {
            let (l, r) = (
                std::mem::take(&mut self.ch_left_pcm),
                std::mem::take(&mut self.ch_right_pcm),
            );
            self.analyzer.process_channels(&l, &r, dt);
            self.ch_left_pcm = l;
            self.ch_right_pcm = r;
        } else {
            // Mix selected, or L/R unavailable: no second and third FFT, and
            // nothing left behind for the renderer to pick up.
            self.analyzer.reset_channels();
        }
    }

    /// Reset all peak hold state to zero.
    fn reset_peaks(&mut self) {
        let n = self.analyzer.bar_count.max(1);
        self.peak_vals        = vec![0.0; n];
        self.peak_hold_timers = vec![0.0; n];
        self.peak_velocities  = vec![0.0; n];
        self.peak_alphas      = vec![1.0; n];
    }

    /// Advance peak hold state by `dt` seconds using the current smoothed magnitudes.
    fn update_peaks(&mut self, dt: f32) {
        if !self.peak_config.enabled { return; }

        // The interval maximum, not the last displayed value: a UI tick can
        // span several cached source rows, and a transient that lived in one
        // the display never showed still has to reach the peak marker. In
        // real-time mode the interval is one frame, so this is the frame.
        let mags = &self.analyzer.peak_input;
        let n    = mags.len();

        if self.peak_vals.len() != n {
            self.peak_vals        = vec![0.0; n];
            self.peak_hold_timers = vec![0.0; n];
            self.peak_velocities  = vec![0.0; n];
            self.peak_alphas      = vec![1.0; n];
        }

        let hold_secs = self.peak_config.hold_ms / 1000.0;

        for i in 0..n {
            let v = mags[i];
            if v >= self.peak_vals[i] {
                self.peak_vals[i]        = v;
                self.peak_hold_timers[i] = 0.0;
                self.peak_velocities[i]  = 0.0;
                self.peak_alphas[i]      = 1.0;
            } else {
                self.peak_hold_timers[i] += dt;
                if self.peak_hold_timers[i] > hold_secs {
                    match self.peak_config.decay_mode {
                        PeakDecayMode::Linear => {
                            self.peak_vals[i] = (self.peak_vals[i]
                                - self.peak_config.fall_speed * dt)
                                .max(v);
                        }
                        PeakDecayMode::Gravity => {
                            self.peak_velocities[i] +=
                                self.peak_config.acceleration * dt;
                            self.peak_vals[i] = (self.peak_vals[i]
                                - self.peak_velocities[i] * dt)
                                .max(v);
                        }
                        PeakDecayMode::FadeOut => {
                            // Exponential decay: visually smooth because brightness
                            // drops quickly at first then eases off naturally.
                            // Scaled by ln(100)≈4.6 so fall_speed produces the same
                            // rough fade duration as Linear mode.
                            let rate = self.peak_config.fall_speed * 4.6;
                            self.peak_alphas[i] *= (-rate * dt).exp();
                            if self.peak_alphas[i] < 0.01 {
                                self.peak_alphas[i] = 0.0;
                                self.peak_vals[i]   = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Render the spectrum as a separate OS window via `show_viewport_immediate`.
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open { return; }

        // Instrumentation: measure the REAL repaint interval (how often this
        // window is actually being redrawn) and the wall-clock cost of the
        // draw. Unlike `current_fps`, which only tracks the FFT throttle,
        // these expose the true CPU driver.
        let show_start = Instant::now();
        if let Some(prev) = self.last_show_time {
            let dt = (show_start - prev).as_secs_f32();
            if dt > 0.0 && dt < 1.0 {
                self.real_repaint_fps = self.real_repaint_fps * 0.9 + (1.0 / dt) * 0.1;
            }
        }
        self.last_show_time = Some(show_start);

        let vp_id = egui::ViewportId::from_hash_of("moosik_spectrum");
        let vp_builder = egui::ViewportBuilder::default()
            .with_title("Spectrum Analyzer")
            .with_inner_size([700.0, 380.0])
            .with_min_inner_size([380.0, 200.0]);

        // `show_viewport_immediate` runs the closure synchronously this frame
        // and lets us capture `&mut self` without any 'static requirement.
        ctx.show_viewport_immediate(vp_id, vp_builder, |vp_ctx, _class| {
            if vp_ctx.input(|i| i.viewport().close_requested()) {
                self.open = false;
                return;
            }
            // F3 toggles debug overlay
            if vp_ctx.input(|i| i.key_pressed(egui::Key::F3)) {
                self.show_debug = !self.show_debug;
            }
            egui::CentralPanel::default().show(vp_ctx, |ui| {
                // ── Row 1: mode + view + channels + loudness ──────────────
                //
                // Wrapped, not clipped. This is one row of about a dozen
                // controls and the window opens at 700 px with a 380 px
                // minimum; a plain `horizontal` runs the tail of it off the
                // right edge with nothing to say it is there. The Channels
                // group was added to the end of it and was simply invisible at
                // the default size — the feature shipped unreachable.
                ui.horizontal_wrapped(|ui| {
                    ui.label("Mode:");
                    ui.selectable_value(&mut self.mode, SpectrumMode::RealTime, "Real-time");
                    ui.selectable_value(&mut self.mode, SpectrumMode::PreProcess, "Pre-process");
                    // Spectro and Octave rely on live FFT norms — unavailable in pre-process
                    // mode (which replays compressed bar data, not raw bins). Auto-fall-back
                    // to Waterfall so the display isn't blank.
                    if self.mode == SpectrumMode::PreProcess
                        && matches!(self.style, VizStyle::Spectrogram | VizStyle::OctaveBands)
                    {
                        self.style = VizStyle::Waterfall;
                    }
                    ui.separator();
                    ui.label("View:");
                    ui.selectable_value(&mut self.style, VizStyle::Bars, "Bars");
                    ui.selectable_value(&mut self.style, VizStyle::Line, "Line");
                    ui.selectable_value(&mut self.style, VizStyle::FilledArea, "Filled");
                    ui.selectable_value(&mut self.style, VizStyle::Waterfall, "Waterfall");
                    // Spectro and Octave are real-time only — hide buttons in pre-process mode.
                    if self.mode == SpectrumMode::RealTime {
                        ui.selectable_value(&mut self.style, VizStyle::Spectrogram, "Spectro");
                        ui.selectable_value(&mut self.style, VizStyle::OctaveBands, "Octave");
                    }
                    ui.selectable_value(&mut self.style, VizStyle::Phasescope, "Phase");
                    ui.separator();
                    // ── Channel view ────────────────────────────────────────
                    // Mix is always offered. The rest are enabled only when a
                    // live two-channel PCM tap is feeding the analyser, and the
                    // reason they are not is on the disabled buttons rather
                    // than discovered by clicking one and seeing nothing change.
                    let ch_reason = self.channel_availability.reason();
                    ui.label("Channels:");
                    for view in channels::ChannelView::ALL {
                        if view == channels::ChannelView::Mix {
                            ui.selectable_value(&mut self.channel_view, view, view.label())
                                .on_hover_text(
                                    "Every channel averaged to one spectrum. \
                                     The default, and the only view the \
                                     pre-processed cache can supply.",
                                );
                            continue;
                        }
                        let enabled = ch_reason.is_none();
                        let btn = ui.add_enabled(
                            enabled,
                            egui::SelectableLabel::new(self.channel_view == view, view.label()),
                        );
                        if enabled {
                            if btn.clicked() {
                                self.channel_view = view;
                            }
                            btn.on_hover_text(match view {
                                channels::ChannelView::Left => "Left channel only.",
                                channels::ChannelView::Right => "Right channel only.",
                                channels::ChannelView::Split =>
                                    "Left above right, sharing one frequency \
                                     and dB scale.",
                                channels::ChannelView::Diff =>
                                    "Per band, the right channel's level minus \
                                     the left channel's, in dB, about a centre \
                                     line at 0 and full-scale at \u{b1}20 dB. \
                                     Magenta means right is louder in that band, \
                                     cyan means left. It is a difference of two \
                                     measured levels, not a subtraction of the \
                                     waveforms and not the M/S Side signal — \
                                     two bands can cancel acoustically and still \
                                     read zero here. Arrangement is under \
                                     Settings.",
                                _ => "Both channels in one plot: cyan is left, \
                                      magenta is right.",
                            });
                        } else if let Some(ref why) = ch_reason {
                            btn.on_hover_text(why.as_str());
                        }
                    }
                    // Shown whenever the group is disabled, not only once a
                    // channel view has been selected — which could not happen,
                    // because selecting one requires the buttons this explains
                    // the absence of. The reason was reachable only by hovering
                    // a greyed-out button, which is not a thing anyone does.
                    if let Some(ref why) = ch_reason {
                        let short = match self.channel_availability {
                            channels::ChannelAvailability::PreProcessMono => "Real-time only",
                            channels::ChannelAvailability::UnsupportedStyle => "Bars/Line/Filled only",
                            channels::ChannelAvailability::Mono => "mono track",
                            channels::ChannelAvailability::Multichannel(n) => {
                                &*format!("{n}-channel track")
                            }
                            _ => "no live PCM",
                        };
                        ui.label(
                            egui::RichText::new(format!("({short})"))
                                .size(11.0)
                                .color(txt_faint(ui.visuals().dark_mode)),
                        )
                        .on_hover_text(why.as_str());
                    }
                    ui.separator();
                    ui.label("Loudness:");
                    let prev_loudness = self.loudness_mode.clone();
                    ui.selectable_value(&mut self.loudness_mode, LoudnessMode::Flat, "Flat");
                    ui.selectable_value(&mut self.loudness_mode, LoudnessMode::EqualLoudness, "ISO 226");
                    if self.loudness_mode != prev_loudness {
                        self.sync_params();
                    }
                    ui.separator();
                    ui.label("Palette:");
                    egui::ComboBox::from_id_salt("spectrum_palette")
                        .selected_text(self.palette_kind.label())
                        .show_ui(ui, |ui| {
                            for p in SpectrumPalette::ALL {
                                let selected = self.palette_kind == p;
                                let clicked = ui.horizontal(|ui| {
                                    paint_palette_swatch(
                                        ui, &Palette::new(p, self.palette_accent),
                                        egui::vec2(34.0, 13.0),
                                    );
                                    ui.selectable_label(selected, p.label()).clicked()
                                }).inner;
                                if clicked { self.palette_kind = p; }
                            }
                        });
                    ui.separator();
                    let dark = ui.visuals().dark_mode;
                    let eq_label = egui::RichText::new("🎛 EQ")
                        .color(if self.show_eq { txt_accent(dark) } else { txt_faint(dark) });
                    if ui.selectable_label(self.show_eq, eq_label)
                        .on_hover_text("Toggle parametric EQ panel.\nClick on spectrum to add bands.\nDrag nodes to adjust.\nRight-click node to remove.")
                        .clicked()
                    {
                        self.show_eq = !self.show_eq;
                    }
                    ui.separator();
                    let dbg_label = egui::RichText::new("🐛 Debug")
                        .color(if self.show_debug { txt_warn(dark) } else { txt_faint(dark) });
                    if ui.selectable_label(self.show_debug, dbg_label)
                        .on_hover_text("Toggle debug overlay (F3)")
                        .clicked()
                    {
                        self.show_debug = !self.show_debug;
                    }
                });

                // ── Row 2: bar count (+ bar gap for Bars) ─────────────────
                ui.horizontal_wrapped(|ui| {
                    ui.label("Bars:");
                    let r = ui.add(egui::Slider::new(&mut self.bar_count,
                            MIN_BAR_COUNT..=MAX_BAR_COUNT)
                        .logarithmic(true).integer().suffix(" bars"));
                    let commit = r.drag_stopped() || (r.changed() && !r.dragged());
                    if commit {
                        self.bar_count = snap_pow2(self.bar_count);
                        self.analyzer.set_bar_count(self.bar_count);
                        self.sync_params();
                        self.try_load_or_flag_reanalysis();
                    }
                    ui.label(format!(
                        "  ({} Hz/bin)",
                        self.analyzer.sample_rate / self.fft_size as u32
                    ));
                    // Bar gap lives here (only meaningful for Bars), so it no
                    // longer needs a row of its own.
                    if self.style == VizStyle::Bars {
                        ui.separator();
                        ui.label("Gap:");
                        ui.add(egui::Slider::new(&mut self.bar_gap, 0.0..=12.0)
                            .step_by(1.0).suffix(" px"))
                            .on_hover_text(
                                "Physical-pixel gap between bars.\n\
                                 0 = no gap (solid fill). Higher values give a more separated look."
                            );
                    }
                });

                // ── Settings sections ─────────────────────────────────────
                // One wrap-around row of toggles instead of three stacked
                // collapsibles, so the plot keeps its height. Each section's
                // body renders full-width below when its toggle is on; the row
                // wraps to more lines only when the window is too narrow.
                ui.horizontal_wrapped(|ui| {
                    let dark = ui.visuals().dark_mode;
                    let mut chip = |open: &mut bool, text: &str| {
                        let col = if *open { txt_accent(dark) } else { txt_dim(dark) };
                        if ui.selectable_label(*open, egui::RichText::new(text).color(col))
                            .clicked()
                        {
                            *open = !*open;
                        }
                    };
                    chip(&mut self.show_fft_settings, "⚙ Analysis");
                    if self.style == VizStyle::Bars {
                        chip(&mut self.show_peak_settings, "📌 Peak Hold");
                    }
                    chip(&mut self.show_art_settings, "🖼 Album Art");
                    if self.mode == SpectrumMode::PreProcess {
                        chip(&mut self.show_cache_settings, "🗄 Cache");
                    }
                });

                // ── FFT params ────────────────────────────────────────────
                if self.show_fft_settings { ui.group(|ui| {
                        let mut rebuild = false;
                        ui.horizontal(|ui| {
                            ui.label("FFT size:");
                            for &sz in &[1024usize, 2048, 4096, 8192, 16384, 32768] {
                                let is_auto = sz == self.auto_fft_size;
                                let sr = self.analyzer.sample_rate.max(1);
                                let ms = sz * 1000 / sr as usize;
                                let text = if is_auto {
                                    format!("{sz} ({ms}ms★)")
                                } else {
                                    format!("{sz} ({ms}ms)")
                                };
                                let has_cache = self.current_path.as_ref().map(|p| {
                                    self.cache_file_set.contains(&cache_path_for(
                                        p, self.bar_count, sz, self.pad_factor, self.overlap,
                                        &self.window_fn, self.min_freq, self.max_freq,
                                        &self.bar_mapping, &self.interp_mode, self.analyzer.dsd_rate, &self.analyzer.aslt_cfg, self.analyzer.pre_fps,
                                    ))
                                }).unwrap_or(false) && self.fft_knobs_apply();
                                let label = if has_cache {
                                    egui::RichText::new(&text).color(txt_ok(ui.visuals().dark_mode))
                                } else if is_auto {
                                    egui::RichText::new(&text).color(txt_warn(ui.visuals().dark_mode))
                                } else {
                                    egui::RichText::new(&text)
                                };
                                let btn = ui.selectable_label(self.fft_size == sz, label);
                                let btn = if is_auto {
                                    btn.on_hover_text("Auto-selected for this sample rate")
                                } else {
                                    btn
                                };
                                if btn.clicked() {
                                    self.fft_size = sz;
                                    self.analyzer.fft_size = sz;
                                    self.auto_fft_size = 0;
                                    rebuild = true;
                                    self.try_load_or_flag_reanalysis();
                                }
                            }
                        });
                        // DSD analysis rate — only meaningful when a DSD track
                        // is loaded. Higher rate = shorter window at the same
                        // FFT size = faster-reacting spectrum (more frames,
                        // bigger cache, a bit more analysis time).
                        if self.current_path.as_deref().is_some_and(crate::dsd::is_dsd_path) {
                            ui.horizontal(|ui| {
                                ui.label("DSD analysis:");
                                let mut changed = false;
                                for &rate in &crate::dsd::decimate::ANALYSIS_RATES {
                                    let name = format!("{:.1} kHz", rate as f32 / 1000.0);
                                    let has = self.current_path.as_ref().map(|p| {
                                        self.cache_file_set.contains(&cache_path_for(
                                            p, self.bar_count, self.fft_size, self.pad_factor,
                                            self.overlap, &self.window_fn, self.min_freq, self.max_freq,
                                            &self.bar_mapping, &self.interp_mode, rate,
                                            &self.analyzer.aslt_cfg, self.analyzer.pre_fps,
                                        ))
                                    }).unwrap_or(false);
                                    let label = if has {
                                        egui::RichText::new(&name).color(txt_ok(ui.visuals().dark_mode))
                                    } else {
                                        egui::RichText::new(&name)
                                    };
                                    let hover = if rate > 176_400 {
                                        "Decimate DSD to 352.8 kHz for analysis — twice the frame rate (faster-reacting spectrum), larger cache."
                                    } else {
                                        "Decimate DSD to 176.4 kHz for analysis — the default; every DSD rate divides into it exactly."
                                    };
                                    if ui.selectable_label(self.analyzer.dsd_rate == rate, label)
                                        .on_hover_text(hover).clicked()
                                        && self.analyzer.dsd_rate != rate {
                                        self.analyzer.dsd_rate = rate;
                                        changed = true;
                                    }
                                }
                                if changed {
                                    // Mirror on_play: the analyzer's world runs at the
                                    // decimated rate, so retime and re-pick the auto FFT.
                                    if let Ok(info) = self.current_path.as_deref()
                                        .ok_or(()).and_then(|p| crate::dsd::parse_file(p).map_err(|_| ())) {
                                        let sr = crate::dsd::decimate::analysis_rate_for(
                                            info.sample_rate, self.analyzer.dsd_rate);
                                        self.analyzer.sample_rate = sr;
                                        if self.auto_fft_size != 0 {
                                            let auto = auto_fft_size_for(sr);
                                            self.fft_size = auto;
                                            self.analyzer.fft_size = auto;
                                            self.auto_fft_size = auto;
                                        }
                                    }
                                    rebuild = true;
                                    self.try_load_or_flag_reanalysis();
                                }
                            });
                        }
                        {
                            let wf_green = |wf: WindowFn| -> egui::RichText {
                                let has = self.current_path.as_ref().map(|p| {
                                    self.cache_file_set.contains(&cache_path_for(
                                        p, self.bar_count, self.fft_size, self.pad_factor,
                                        self.overlap, &wf, self.min_freq, self.max_freq,
                                        &self.bar_mapping, &self.interp_mode, self.analyzer.dsd_rate, &self.analyzer.aslt_cfg, self.analyzer.pre_fps,
                                    ))
                                }).unwrap_or(false) && self.fft_knobs_apply();
                                let name = match wf {
                                    WindowFn::Hann    => "Hann",
                                    WindowFn::Hamming => "Hamming",
                                    WindowFn::Blackman => "Blackman",
                                    WindowFn::FlatTop  => "Flat-top",
                                };
                                if has { egui::RichText::new(name).color(txt_ok(ui.visuals().dark_mode)) }
                                else   { egui::RichText::new(name) }
                            };
                            let lbl_hann    = wf_green(WindowFn::Hann);
                            let lbl_hamming = wf_green(WindowFn::Hamming);
                            let lbl_black   = wf_green(WindowFn::Blackman);
                            let lbl_flat    = wf_green(WindowFn::FlatTop);
                            ui.horizontal(|ui| {
                                ui.label("Window:");
                                let prev = self.window_fn.clone();
                                ui.selectable_value(&mut self.window_fn, WindowFn::Hann, lbl_hann)
                                    .on_hover_text("Raised cosine. Best general-purpose window — good sidelobe rejection with minimal smearing. Default.");
                                ui.selectable_value(&mut self.window_fn, WindowFn::Hamming, lbl_hamming)
                                    .on_hover_text("Optimised for the first sidelobe only. Slightly sharper main lobe than Hann, but higher distant sidelobes.");
                                ui.selectable_value(&mut self.window_fn, WindowFn::Blackman, lbl_black)
                                    .on_hover_text("Three-term cosine sum. Excellent sidelobe suppression at the cost of a wider main lobe (less frequency resolution).");
                                ui.selectable_value(&mut self.window_fn, WindowFn::FlatTop, lbl_flat)
                                    .on_hover_text("Near-unity passband — amplitude error < 0.01 dB. Wide main lobe, so poor frequency resolution. Use only for level measurement.");
                                if self.window_fn != prev {
                                    self.analyzer.window_fn = self.window_fn.clone();
                                    rebuild = true;
                                    self.try_load_or_flag_reanalysis();
                                }
                            });
                        }
                        {
                            let im_green = |im: InterpolationMode| -> egui::RichText {
                                let has = self.current_path.as_ref().map(|p| {
                                    self.cache_file_set.contains(&cache_path_for(
                                        p, self.bar_count, self.fft_size, self.pad_factor,
                                        self.overlap, &self.window_fn, self.min_freq, self.max_freq,
                                        &self.bar_mapping, &im, self.analyzer.dsd_rate, &self.analyzer.aslt_cfg, self.analyzer.pre_fps,
                                    ))
                                }).unwrap_or(false) && self.fft_knobs_apply();
                                let name = match im {
                                    InterpolationMode::None      => "None",
                                    InterpolationMode::Linear    => "Linear",
                                    InterpolationMode::CatmullRom => "Catmull-Rom",
                                    InterpolationMode::Pchip     => "PCHIP",
                                    InterpolationMode::Akima     => "Akima",
                                    InterpolationMode::Lanczos   => "Lanczos",
                                };
                                if has { egui::RichText::new(name).color(txt_ok(ui.visuals().dark_mode)) }
                                else   { egui::RichText::new(name) }
                            };
                            let lbl_none   = im_green(InterpolationMode::None);
                            let lbl_linear = im_green(InterpolationMode::Linear);
                            let lbl_cr     = im_green(InterpolationMode::CatmullRom);
                            let lbl_pchip  = im_green(InterpolationMode::Pchip);
                            let lbl_akima  = im_green(InterpolationMode::Akima);
                            let lbl_lanc   = im_green(InterpolationMode::Lanczos);
                            ui.horizontal(|ui| {
                                ui.label("Interpolation:");
                                let prev = self.interp_mode.clone();
                                ui.selectable_value(&mut self.interp_mode, InterpolationMode::None, lbl_none)
                                    .on_hover_text("Nearest bin — no interpolation. Most honest to the raw FFT.");
                                ui.selectable_value(&mut self.interp_mode, InterpolationMode::Linear, lbl_linear)
                                    .on_hover_text("Linear blend between adjacent bins. Fast but angular in the low end.");
                                ui.selectable_value(&mut self.interp_mode, InterpolationMode::CatmullRom, lbl_cr)
                                    .on_hover_text("Smooth cubic spline. Good balance of quality and speed.");
                                ui.selectable_value(&mut self.interp_mode, InterpolationMode::Pchip, lbl_pchip)
                                    .on_hover_text("Monotone cubic (Fritsch-Carlson) — no overshoot, shape-preserving.");
                                ui.selectable_value(&mut self.interp_mode, InterpolationMode::Akima, lbl_akima)
                                    .on_hover_text("Local cubic designed for scientific data. Smooth without oscillation.");
                                ui.selectable_value(&mut self.interp_mode, InterpolationMode::Lanczos, lbl_lanc)
                                    .on_hover_text("Sinc-windowed (a=3). Best accuracy but can ring near sharp peaks.");
                                if self.interp_mode != prev {
                                    self.analyzer.interp_mode = self.interp_mode.clone();
                                    self.try_load_or_flag_reanalysis();
                                }
                            });
                        }
                        ui.horizontal(|ui| {
                            ui.label("Zero-padding:");
                            let prev_pad = self.pad_factor;
                            for &pf in &[1usize, 2, 4, 8, 16, 32, 64] {
                                let label_text = if pf == 1 { "1× (off)".to_string() } else { format!("{}×", pf) };
                                let has_cache = self.current_path.as_ref().map(|p| {
                                    let candidate = cache_path_for(
                                        p, self.bar_count, self.fft_size, pf, self.overlap,
                                        &self.window_fn, self.min_freq, self.max_freq,
                                        &self.bar_mapping, &self.interp_mode, self.analyzer.dsd_rate, &self.analyzer.aslt_cfg, self.analyzer.pre_fps,
                                    );
                                    self.cache_file_set.contains(&candidate)
                                }).unwrap_or(false) && self.fft_knobs_apply();
                                let rich = if has_cache {
                                    egui::RichText::new(label_text).color(txt_ok(ui.visuals().dark_mode))
                                } else {
                                    egui::RichText::new(label_text)
                                };
                                let btn = ui.selectable_label(self.pad_factor == pf, rich);
                                let btn = match pf {
                                    1  => btn.on_hover_text("No padding — rely entirely on \
                                          interpolation. Padding never adds resolution; it \
                                          samples the same transform more finely."),
                                    2  => btn.on_hover_text("2× denser bins. Good default."),
                                    4  => btn.on_hover_text("4× — near-ideal for bar visualization."),
                                    8  => btn.on_hover_text("8× — visually indistinguishable from interpolation; interpolation almost irrelevant."),
                                    16 => btn.on_hover_text("16× — interpolation fully redundant; pre-process recommended."),
                                    32 => btn.on_hover_text("32× — pre-process only."),
                                    64 => btn.on_hover_text("64× — maximum sinc fidelity. Pre-process only."),
                                    _  => btn,
                                };
                                if btn.clicked() {
                                    self.pad_factor = pf;
                                    self.analyzer.pad_factor = pf;
                                    rebuild = true;
                                }
                            }
                            if self.pad_factor != prev_pad {
                                self.try_load_or_flag_reanalysis();
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Overlap:");
                            for &(text, val) in &[("50%", 0.5f32), ("75%", 0.75), ("87.5%", 0.875)] {
                                let has_cache = self.current_path.as_ref().map(|p| {
                                    self.cache_file_set.contains(&cache_path_for(
                                        p, self.bar_count, self.fft_size, self.pad_factor, val,
                                        &self.window_fn, self.min_freq, self.max_freq,
                                        &self.bar_mapping, &self.interp_mode, self.analyzer.dsd_rate, &self.analyzer.aslt_cfg, self.analyzer.pre_fps,
                                    ))
                                }).unwrap_or(false) && self.fft_knobs_apply();
                                let rich = if has_cache {
                                    egui::RichText::new(text).color(txt_ok(ui.visuals().dark_mode))
                                } else {
                                    egui::RichText::new(text)
                                };
                                let prev = self.overlap;
                                let btn = ui.selectable_label((self.overlap - val).abs() < 0.01, rich);
                                let btn = match (val * 1000.0) as u32 {
                                    500 => btn.on_hover_text("50% — standard STFT. Fastest analysis."),
                                    750 => btn.on_hover_text("75% — 2× more frames, smoother temporal detail."),
                                    _   => btn.on_hover_text("87.5% — 4× more frames. Maximum temporal smoothness."),
                                };
                                if btn.clicked() {
                                    self.overlap = val;
                                    self.analyzer.overlap = val;
                                    if (self.overlap - prev).abs() > 0.001 {
                                        self.try_load_or_flag_reanalysis();
                                    }
                                }
                            }
                        });
                        {
                            let bm_green = |bm: BarMappingMode| -> egui::RichText {
                                let has = self.current_path.as_ref().map(|p| {
                                    self.cache_file_set.contains(&cache_path_for(
                                        p, self.bar_count, self.fft_size, self.pad_factor,
                                        self.overlap, &self.window_fn, self.min_freq, self.max_freq,
                                        &bm, &self.interp_mode, self.analyzer.dsd_rate, &self.analyzer.aslt_cfg, self.analyzer.pre_fps,
                                    ))
                                }).unwrap_or(false);
                                let name = match bm {
                                    BarMappingMode::FlatOverlap => "Flat",
                                    BarMappingMode::Gaussian    => "Gaussian",
                                    BarMappingMode::Cqt         => "CQT",
                                    BarMappingMode::Superlet    => "Superlet",
                                };
                                if has { egui::RichText::new(name).color(txt_ok(ui.visuals().dark_mode)) }
                                else   { egui::RichText::new(name) }
                            };
                            let lbl_flat  = bm_green(BarMappingMode::FlatOverlap);
                            let lbl_gauss = bm_green(BarMappingMode::Gaussian);
                            let lbl_cqt   = bm_green(BarMappingMode::Cqt);
                            let lbl_slt   = bm_green(BarMappingMode::Superlet);
                            if !self.fft_knobs_apply() {
                                ui.label(egui::RichText::new(
                                    "Superlet analyses the waveform directly — FFT size, window, \
                                     padding, overlap and interpolation do not apply. Zero-padding \
                                     adds no resolution here: it interpolates FFT bins, and there \
                                     are none. Resolution comes from the preset and bar count.")
                                    .size(10.0).color(txt_faint(ui.visuals().dark_mode)));
                            }
                            ui.horizontal(|ui| {
                                ui.label("Bar mapping:");
                                let prev = self.bar_mapping.clone();
                                ui.selectable_value(&mut self.bar_mapping, BarMappingMode::FlatOverlap, lbl_flat)
                                    .on_hover_text("Equal weight to all FFT bins within the bar's frequency range.");
                                ui.selectable_value(&mut self.bar_mapping, BarMappingMode::Gaussian, lbl_gauss)
                                    .on_hover_text("Bins near the bar's centre frequency weighted more heavily. More natural.");
                                ui.selectable_value(&mut self.bar_mapping, BarMappingMode::Cqt, lbl_cqt)
                                    .on_hover_text("Constant-Q Transform — Hann kernel with bandwidth ∝ frequency.\nEach bar has identical relative frequency resolution. Best for music.");
                                ui.selectable_value(&mut self.bar_mapping, BarMappingMode::Superlet, lbl_slt)
                                    .on_hover_text(format!(
                                        "Adaptive Superlet Transform — a direct wavelet analysis, not an FFT mapping.\n\
                                         Highest resolution available, at minutes of pre-processing per track.\n\
                                         Pre-process only; live view keeps using the FFT.\n\n\
                                         Current: {} — bass window {:.2} s, grid Q {:.0} @ {:.0} fps",
                                        self.analyzer.aslt_preset.map(|p| p.label()).unwrap_or("Custom"),
                                        aslt::window_seconds_at(
                                            self.min_freq,
                                            aslt::grid_q(self.bar_count, self.min_freq, self.max_freq),
                                            &self.analyzer.aslt_cfg,
                                        ),
                                        aslt::grid_q(self.bar_count, self.min_freq, self.max_freq),
                                        self.analyzer.pre_fps,
                                    ));
                                if self.bar_mapping != prev {
                                    self.analyzer.bar_mapping = self.bar_mapping.clone();
                                    self.try_load_or_flag_reanalysis();
                                }
                            });

                            // Superlet quality ladder — only meaningful while
                            // Superlet is the active mapping, so it stays hidden
                            // otherwise rather than sitting there greyed out.
                            if self.bar_mapping == BarMappingMode::Superlet {
                                // The grid Q the readouts quote has to be the
                                // one at the frequency being quoted: under ERB
                                // it varies along the spectrum, and a single
                                // number would describe a display that is not
                                // on screen.
                                let n_bars = self.bar_count;
                                let (lo_hz, hi_hz) = (self.min_freq, self.max_freq);
                                let grid_at_hz = |sc: freq_scale::FreqScale, f: f32| {
                                    // Straight inversion, not a search. Scanning
                                    // every bar for the nearest centre meant a
                                    // thousand `bar_center` calls per lookup, and
                                    // on a blended scale each of those is a
                                    // bisection — about a third of a million
                                    // evaluations per frame with this panel open,
                                    // which is what dropped the frame rate while
                                    // it was.
                                    let t = sc.position_of(f, lo_hz, hi_hz);
                                    let bar = (t * n_bars as f32 - 0.5)
                                        .round()
                                        .clamp(0.0, n_bars.saturating_sub(1) as f32)
                                        as usize;
                                    sc.grid_q_at(bar, n_bars, lo_hz, hi_hz)
                                };
                                // What the transform is competing against, so the
                                // labels can say "better than the FFT" and mean it.
                                let fft_sigma = aslt::fft_sigma_hz(
                                    self.analyzer.sample_rate, self.fft_size);
                                let mut retune: Option<(Option<aslt::AsltPreset>, aslt::AsltConfig)> = None;
                                ui.horizontal(|ui| {
                                    ui.label("Quality:");
                                    for p in [
                                        aslt::AsltPreset::Fast, aslt::AsltPreset::Standard,
                                        aslt::AsltPreset::High, aslt::AsltPreset::Ultra,
                                        aslt::AsltPreset::Extreme,
                                    ] {
                                        let cfg = p.config();
                                        let selected = self.analyzer.aslt_preset == Some(p);
                                        let beats_fft = 60.0 / aslt::effective_q_at(60.0, grid_at_hz(cfg.scale, 60.0), &cfg)
                                            < fft_sigma;
                                        if ui.selectable_label(selected, p.label())
                                            .on_hover_text(format!(
                                                "Bass window {:.2} s, Q {:.0} at 60 Hz.\n{}",
                                                aslt::window_seconds_at(60.0, grid_at_hz(cfg.scale, 60.0), &cfg),
                                                aslt::effective_q_at(60.0, grid_at_hz(cfg.scale, 60.0), &cfg),
                                                if beats_fft {
                                                    "Resolves bass detail the FFT cannot reach."
                                                } else {
                                                    "Below FFT resolution in the bass; still much faster in the treble."
                                                },
                                            ))
                                            .clicked() && !selected
                                        {
                                            retune = Some((Some(p), cfg));
                                        }
                                    }
                                    let custom = self.analyzer.aslt_preset.is_none();
                                    if ui.selectable_label(custom, "Custom")
                                        .on_hover_text("Set the window budget and wavelet spread by hand.")
                                        .clicked() && !custom
                                    {
                                        retune = Some((None, self.analyzer.aslt_cfg.clone()));
                                    }
                                });
                                if self.analyzer.aslt_preset.is_none() {
                                    let mut cfg = self.analyzer.aslt_cfg.clone();
                                    let before = cfg.clone();
                                    ui.horizontal(|ui| {
                                        ui.label("Max window:");
                                        ui.add(egui::Slider::new(&mut cfg.max_window_s, 0.1..=8.0)
                                            .logarithmic(true).suffix(" s"))
                                            .on_hover_text(
                                                "Longest analysis window. This is the bass trade: \
                                                 constant-Q at 20 Hz needs ~10 s, so whatever you \
                                                 set here is where bass detail stops. Never binds \
                                                 above ~1 kHz.");
                                        // The same setting stated as what it
                                        // actually decides. Both are live and
                                        // either can be driven — the seconds are
                                        // the cost, the hertz are the result,
                                        // and which one someone thinks in is
                                        // their business.
                                        ui.label("= full detail above:");
                                        let mut hz = aslt::full_detail_above(grid_at_hz(cfg.scale, 1000.0), &cfg);
                                        if ui.add(egui::DragValue::new(&mut hz)
                                            .range(30.0..=8000.0).speed(5.0).suffix(" Hz"))
                                            .on_hover_text(
                                                "The lowest frequency that still lands in a single \
                                                 bar. Below it the window cap binds and a tone \
                                                 spreads over roughly grid-Q ÷ its own Q bars — \
                                                 that spreading is the uncertainty principle, not \
                                                 a defect, and no setting removes it. Driving this \
                                                 sets the window above.")
                                            .changed()
                                        {
                                            cfg.max_window_s = aslt::window_for_full_detail_above(
                                                hz, grid_at_hz(cfg.scale, hz), &cfg,
                                            ).clamp(0.1, 8.0);
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Sharpness:");
                                        ui.add(egui::Slider::new(&mut cfg.q_ratio, 0.25..=2.0)
                                            .suffix("× grid"))
                                            .on_hover_text(
                                                "Target Q relative to the bar grid. 1.0 resolves \
                                                 exactly what the bars can draw; above that is \
                                                 detail the display cannot show.");
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Wavelets:");
                                        ui.add(egui::Slider::new(&mut cfg.n_wavelets, 1..=9))
                                            .on_hover_text(
                                                "Members per superlet. 1 is a plain wavelet transform \
                                                 — sharpest peak, no cross-checking. More suppress \
                                                 energy the set disagrees about.");
                                        ui.label("Spread:");
                                        ui.add(egui::Slider::new(&mut cfg.spread, 0.0..=0.95))
                                            .on_hover_text(
                                                "Shortest member as a fraction of the longest. 0 is \
                                                 the paper's layout and costs 1.73× resolution for \
                                                 the same time; ~0.5 keeps most of the cross-checking \
                                                 for 1.31×.");
                                    });
                                    let (lens_hz, lens_oct, lens_gain) =
                                        cfg.scale.lens().unwrap_or((1000.0, 2.0, 0.0));
                                    ui.horizontal(|ui| {
                                        ui.label("Bass width:");
                                        let mut tilt = cfg.scale.tilt();
                                        // Reversed range so dragging right
                                        // widens the bass, which is the
                                        // direction the label promises.
                                        let r = ui.add(
                                            egui::Slider::new(&mut tilt, 1.0..=-1.0)
                                                .step_by(0.05)
                                                .custom_formatter(|v, _| match v {
                                                    v if v == 0.0 => "log".into(),
                                                    v if v == 1.0 => "ERB".into(),
                                                    v if v > 0.0 => format!("-{:.0}%", v * 100.0),
                                                    v => format!("+{:.0}%", -v * 100.0),
                                                }))
                                            .on_hover_text(
                                                "How much of the display the bass gets. 'log' is \
                                                 constant bars per octave, the classic analyser \
                                                 axis. 'ERB' is constant bars per auditory filter \
                                                 (Glasberg & Moore 1990) — the bass narrows and \
                                                 the treble opens up. Past log, the bass keeps \
                                                 widening.\n\nThis is taste, not accuracy. A tone's \
                                                 blur and a bassline's travel both scale with bar \
                                                 density, so the ratio between them is the same \
                                                 everywhere on this slider — what changes is how \
                                                 much screen the bass gets to move across. And \
                                                 while the window cap is binding down there, \
                                                 which it is at any setting below about 1 s, \
                                                 widening the bass costs no extra compute at all.\
                                                 \n\nEach setting caches separately, so moving \
                                                 back and forth is free after the first analysis.");
                                        if r.changed() {
                                            cfg.scale = freq_scale::FreqScale::from_parts(
                                                tilt, lens_hz, lens_oct, lens_gain);
                                        }
                                        if ui.small_button("log").clicked() {
                                            cfg.scale = freq_scale::FreqScale::from_parts(
                                                0.0, lens_hz, lens_oct, lens_gain);
                                        }
                                        if ui.small_button("ERB").clicked() {
                                            cfg.scale = freq_scale::FreqScale::from_parts(
                                                1.0, lens_hz, lens_oct, lens_gain);
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Zoom:");
                                        let mut on = lens_gain > 0.0;
                                        if ui.checkbox(&mut on, "").changed() {
                                            cfg.scale = freq_scale::FreqScale::from_parts(
                                                cfg.scale.tilt(), lens_hz, lens_oct,
                                                if on { 2.0 } else { 0.0 },
                                            );
                                        }
                                        let mut hz = lens_hz;
                                        let mut oct = lens_oct;
                                        let mut gain = lens_gain;
                                        let e = ui.add_enabled_ui(on, |ui| {
                                            let a = ui.add(egui::DragValue::new(&mut hz)
                                                .range(20.0..=20_000.0).speed(10.0).suffix(" Hz"))
                                                .on_hover_text("Where the extra detail goes.");
                                            let b = ui.add(egui::Slider::new(&mut oct, 0.25..=6.0)
                                                .suffix(" oct").text(""))
                                                .on_hover_text("How wide the magnified region is.");
                                            let c = ui.add(egui::Slider::new(&mut gain, 0.5..=8.0)
                                                .text("×"))
                                                .on_hover_text(
                                                    "How much extra detail at the centre. Bars are \
                                                     taken from the rest of the spectrum to pay \
                                                     for it — the count never changes.");
                                            a.changed() || b.changed() || c.changed()
                                        }).inner;
                                        if e {
                                            cfg.scale = freq_scale::FreqScale::from_parts(
                                                cfg.scale.tilt(), hz, oct, gain);
                                        }
                                    });
                                    // Zooming above the bass is not free: nothing
                                    // caps the window up there, so packing bars in
                                    // raises the resolution actually demanded.
                                    // Quoted as a ratio against the same settings
                                    // with the lens off, because a multiplier on a
                                    // twelve-minute analysis is the part that
                                    // matters.
                                    if cfg.scale.lens().is_some() {
                                        let mut flat = cfg.clone();
                                        flat.scale =
                                            freq_scale::FreqScale::from_tilt(cfg.scale.tilt());
                                        let cost = |c: &aslt::AsltConfig| -> f64 {
                                            aslt::bar_taps_per_frame(
                                                self.analyzer.sample_rate, self.bar_count,
                                                self.min_freq, self.max_freq, c,
                                            ).iter().sum()
                                        };
                                        let (a, b) = (cost(&cfg), cost(&flat));
                                        let ratio = if b > 0.0 { a / b } else { 1.0 };
                                        let msg = format!("zoom costs {ratio:.2}x the analysis time");
                                        let col = if ratio > 1.5 {
                                            txt_warn(ui.visuals().dark_mode)
                                        } else {
                                            txt_dim(ui.visuals().dark_mode)
                                        };
                                        ui.label(egui::RichText::new(msg).size(10.0).color(col));
                                    }
                                    ui.label(egui::RichText::new(format!(
                                        "60 Hz: σ {:.2} Hz over {:.2} s  ·  FFT gives σ {fft_sigma:.2} Hz over {:.0} ms  ·  10 kHz: {:.0} ms",
                                        60.0 / aslt::effective_q_at(60.0, grid_at_hz(cfg.scale, 60.0), &cfg),
                                        aslt::window_seconds_at(60.0, grid_at_hz(cfg.scale, 60.0), &cfg),
                                        1000.0 * self.fft_size as f32 / self.analyzer.sample_rate as f32,
                                        1000.0 * aslt::window_seconds_at(10_000.0, grid_at_hz(cfg.scale, 10_000.0), &cfg),
                                    )).size(10.0).color(txt_dim(ui.visuals().dark_mode)));
                                    // How wide a single bass tone will actually
                                    // draw, which is the thing people notice
                                    // first and the hardest to predict from a
                                    // window length in seconds.
                                    let spread_at = |f: f32| {
                                        let g = grid_at_hz(cfg.scale, f);
                                        (g / aslt::effective_q_at(f, g, &cfg)).max(1.0)
                                    };
                                    ui.label(egui::RichText::new(format!(
                                        "a pure tone covers ≈{:.0} bars at 60 Hz, ≈{:.0} at 200 Hz, 1 bar above {:.0} Hz",
                                        spread_at(60.0), spread_at(200.0),
                                        aslt::full_detail_above(grid_at_hz(cfg.scale, 1000.0), &cfg),
                                    )).size(10.0).color(txt_dim(ui.visuals().dark_mode)));
                                    if cfg != before { retune = Some((None, cfg)); }
                                }
                                ui.horizontal(|ui| {
                                    ui.label("Frame rate:");
                                    let mut fps = self.analyzer.pre_fps;
                                    if ui.add(egui::Slider::new(&mut fps, 30.0..=240.0).step_by(30.0).suffix(" fps"))
                                        .on_hover_text("Pre-processed frames per second. Cost scales linearly, and so does cache size.")
                                        .changed()
                                    {
                                        self.analyzer.pre_fps = fps;
                                        self.try_load_or_flag_reanalysis();
                                    }
                                });
                                self.compute_budget_ui(ui);
                                if let Some((preset, cfg)) = retune {
                                    self.analyzer.aslt_preset = preset;
                                    self.analyzer.aslt_cfg = cfg;
                                    self.try_load_or_flag_reanalysis();
                                }
                            }
                        }
                        ui.horizontal(|ui| {
                            // One control, whichever mode is showing — but two
                            // values, persisted apart. They are not the same
                            // quantity: Real-time is retention per accepted
                            // tick, Pre-process is retention at a 60 Hz
                            // reference converted for the source interval.
                            // Sharing one number made a fast display change what
                            // Real-time meant.
                            let pre = self.mode == SpectrumMode::PreProcess;
                            ui.label(if pre { "Smoothing (cached):" } else { "Smoothing (live):" });
                            let value = if pre { &mut self.pre_smoothing } else { &mut self.smoothing };
                            let r = ui.add(
                                egui::Slider::new(value, 0.0..=0.97).step_by(0.01),
                            );
                            // Quoted in milliseconds, because that is the thing
                            // being chosen and the retention figure hides it:
                            // 0.75 is 174 ms, seven times what 0.125 gives, and
                            // nothing on screen used to say so.
                            let hint = if pre {
                                let v = self.pre_smoothing;
                                if v <= 0.0 {
                                    "Off — each cached row is shown exactly as analysed.".to_string()
                                } else {
                                    format!(
                                        "Cached playback. A step reaches 95% in {:.0} ms, the same \
                                         on every machine and at every analysis rate. Default {:.3} \
                                         is {:.0} ms.",
                                        timing::step_response_95_secs(v) * 1000.0,
                                        timing::DEFAULT_PRE_SMOOTHING,
                                        timing::step_response_95_secs(timing::DEFAULT_PRE_SMOOTHING)
                                            * 1000.0,
                                    )
                                }
                            } else {
                                "Live analysis. Retention per analyser tick, so the response \
                                 follows Max FPS — the behaviour this control has always had."
                                    .to_string()
                            };
                            r.on_hover_text(hint);
                            self.analyzer.smoothing = self.smoothing;
                            self.analyzer.pre_smoothing = self.pre_smoothing;
                        });
                        if self.channel_view == channels::ChannelView::Diff {
                            ui.horizontal(|ui| {
                                ui.label("Diff layout:");
                                for o in [
                                    channels::DiffOrientation::Horizontal,
                                    channels::DiffOrientation::Vertical,
                                ] {
                                    ui.selectable_value(
                                        &mut self.diff_layout.orientation,
                                        o,
                                        o.label(),
                                    );
                                }
                                ui.separator();
                                let (pos, neg) = self.diff_layout.end_labels();
                                let swap = if self.diff_layout.orientation
                                    == channels::DiffOrientation::Vertical
                                {
                                    format!("{neg} ◀ ▶ {pos}")
                                } else {
                                    format!("{pos} ▲ ▼ {neg}")
                                };
                                ui.checkbox(&mut self.diff_layout.flip_channels, "Swap L/R")
                                    .on_hover_text(format!(
                                        "Which channel sits on which side. Currently {swap}. \
                                         Convention, not fact — the label moves with it, so the \
                                         plot cannot end up saying the opposite of what it shows."
                                    ));
                                ui.checkbox(&mut self.diff_layout.flip_frequency, "Flip freq")
                                    .on_hover_text(
                                        "Run the frequency axis the other way — high to low \
                                         instead of low to high, or bottom to top instead of \
                                         top to bottom.",
                                    );
                            });
                        }
                        ui.horizontal(|ui| {
                            ui.label("Waterfall history:");
                            let r = ui.add(
                                egui::Slider::new(
                                    &mut self.waterfall_secs,
                                    timing::WATERFALL_SECS_RANGE,
                                )
                                .step_by(0.01)
                                .suffix(" s"),
                            );
                            // The row count is what it costs, and it follows
                            // the row rate rather than the slider, so it is
                            // worth showing rather than leaving to be guessed.
                            // So is whether that rate is measured or requested,
                            // and whether a bound moved the span off the slider.
                            let (hz, nominal) = self.waterfall_row_rate();
                            let rows = timing::waterfall_rows_for(self.waterfall_secs, hz);
                            let span = timing::waterfall_span_secs(rows, hz);
                            let bounded = (span - self.waterfall_secs as f64).abs() > 0.01;
                            let mut shown = if nominal {
                                format!("{rows} rows @ up to {hz:.0}/s (nominal)")
                            } else {
                                format!("{rows} rows @ {hz:.0}/s")
                            };
                            if bounded {
                                let which = if rows <= timing::WATERFALL_MIN_ROWS {
                                    "min"
                                } else {
                                    "max"
                                };
                                shown.push_str(&format!(
                                    " — {which} {rows} rows, so {span:.2} s"
                                ));
                            }
                            ui.label(
                                egui::RichText::new(shown)
                                    .size(10.0)
                                    .color(txt_faint(ui.visuals().dark_mode)),
                            );
                            r.on_hover_text(if nominal {
                                "How much of the rolling waterfall's past to keep. \
                                 The waterfall advances once per analysis update and \
                                 the ring is a fixed number of rows, so the span is \
                                 rows divided by the rate they arrive at.\n\n\
                                 In Real-time that rate is taken from Max FPS, which \
                                 is a ceiling the analyser is asked to respect rather \
                                 than a rate anything has measured — so this duration \
                                 is nominal. If the machine delivers fewer updates a \
                                 second than Max FPS asks for, the same rows cover \
                                 more time and the history reaches further back than \
                                 the slider says.\n\n\
                                 The ring is held between 32 and 2048 rows. Where a \
                                 bound binds it, the span it actually covers is shown \
                                 beside the slider instead of the value requested."
                            } else {
                                "How much of the rolling waterfall's past to keep. \
                                 The waterfall advances once per cached row consumed \
                                 and the ring is a fixed number of rows, so the span \
                                 is rows divided by the cache's frame rate. Every \
                                 cached row is consumed, so this duration is measured \
                                 rather than nominal.\n\n\
                                 The ring is held between 32 and 2048 rows. Where a \
                                 bound binds it, the span it actually covers is shown \
                                 beside the slider instead of the value requested."
                            });
                        });
                        ui.horizontal(|ui| {
                            ui.label("Min Hz:");
                            let r = ui.add(egui::DragValue::new(&mut self.min_freq)
                                .range(10.0..=500.0).speed(1.0).suffix(" Hz"));
                            ui.label("Max Hz:");
                            let r2 = ui.add(egui::DragValue::new(&mut self.max_freq)
                                .range(1000.0..=24000.0).speed(10.0).suffix(" Hz"));
                            if r.changed() || r2.changed() {
                                self.sync_params();
                                self.try_load_or_flag_reanalysis();
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Max FPS:");
                            ui.add(egui::Slider::new(&mut self.max_fps, 1.0..=240.0)
                                .step_by(1.0).suffix(" fps"));
                        });
                        if rebuild {
                            self.analyzer.rebuild_fft();
                        }
                    }); }

                // ── Pre-process cache ─────────────────────────────────────
                // Warning banner and the stats refresh stay live; the actual
                // management controls tuck into the 🗄 Cache chip so they don't
                // eat plot height when unused.
                if self.mode == SpectrumMode::PreProcess {
                    // Banner when settings changed and no matching cache exists
                    if self.needs_reanalysis && self.current_path.is_some() {
                        ui.horizontal(|ui| {
                            let analyzing = self.analyzer.is_analyzing.load(Ordering::Relaxed);
                            let label = if analyzing {
                                "⚠ No cache for current settings — waiting for analysis to finish…"
                            } else {
                                "⚠ No cache for current settings."
                            };
                            ui.label(egui::RichText::new(label)
                                .size(11.0).color(txt_warn(ui.visuals().dark_mode)));
                            let has_path = self.current_path.is_some();
                            if ui.add_enabled(has_path && !analyzing,
                                egui::Button::new("🔄 Re-analyze now")).clicked()
                                && let Some(ref p) = self.current_path.clone() {
                                let cache = cache_path_for(p, self.bar_count, self.fft_size, self.pad_factor, self.overlap, &self.window_fn, self.min_freq, self.max_freq, &self.bar_mapping, &self.interp_mode, self.analyzer.dsd_rate, &self.analyzer.aslt_cfg, self.analyzer.pre_fps);
                                let _ = std::fs::remove_file(&cache);
                                self.analyzer.clear_pre_frames();
                                self.analyzer.start_preprocess(p.clone());
                                self.needs_reanalysis = false;
                                self.status_msg = String::new();
                            }
                        });
                    }

                    // Refresh cache stats + file set (no UI). Must run even when
                    // the Cache chip is closed — the FFT-size / window / padding
                    // buttons colour green from cache_file_set to show which
                    // combinations are already analysed.
                    let stale = self.cache_stats_at
                        .map(|t| t.elapsed().as_secs_f32() > 2.0)
                        .unwrap_or(true);
                    if stale {
                        self.cache_stats = cache_dir_stats();
                        let dir = home_dir().join(".moosik").join("cache");
                        self.cache_file_set = std::fs::read_dir(&dir)
                            .into_iter().flatten().filter_map(|e| e.ok())
                            .filter(|e| e.path().extension().map(|x| x == "spectrumcache").unwrap_or(false))
                            .map(|e| e.path())
                            .collect();
                        self.cache_stats_at = Some(Instant::now());
                    }

                    if self.show_cache_settings { ui.group(|ui| {
                        ui.horizontal(|ui| {
                            let analyzing = self.analyzer.is_analyzing.load(Ordering::Relaxed);
                            let has_path  = self.current_path.is_some();
                            if ui.add_enabled(has_path && !analyzing,
                                egui::Button::new("🗑 Clear Cache")).clicked()
                                && let Some(ref p) = self.current_path.clone() {
                                let cache = cache_path_for(p, self.bar_count, self.fft_size, self.pad_factor, self.overlap, &self.window_fn, self.min_freq, self.max_freq, &self.bar_mapping, &self.interp_mode, self.analyzer.dsd_rate, &self.analyzer.aslt_cfg, self.analyzer.pre_fps);
                                let existed = cache.exists();
                                let _ = std::fs::remove_file(&cache);
                                self.analyzer.clear_pre_frames();
                                self.needs_reanalysis = false;
                                self.cache_stats_at = None;
                                self.status_msg = if existed {
                                    "Cache cleared.".into()
                                } else {
                                    "No cache file found.".into()
                                };
                            }
                            if ui.add_enabled(has_path && !analyzing,
                                egui::Button::new("🔄 Re-analyze")).clicked()
                                && let Some(ref p) = self.current_path.clone() {
                                let cache = cache_path_for(p, self.bar_count, self.fft_size, self.pad_factor, self.overlap, &self.window_fn, self.min_freq, self.max_freq, &self.bar_mapping, &self.interp_mode, self.analyzer.dsd_rate, &self.analyzer.aslt_cfg, self.analyzer.pre_fps);
                                let _ = std::fs::remove_file(&cache);
                                self.analyzer.clear_pre_frames();
                                self.analyzer.start_preprocess(p.clone());
                                self.needs_reanalysis = false;
                                self.status_msg = String::new();
                            }
                            if analyzing {
                                let pct = self.analyzer.analysis_progress.load(Ordering::Relaxed);
                                let eta = self.analyzer.eta_secs.load(Ordering::Relaxed);
                                ui.spinner();
                                // The ETA lives here rather than only on the plot
                                // overlay: that overlay is gated on the plot being
                                // silent, so during normal PCM playback — the live
                                // FFT still drawing — it never appears at all.
                                ui.label(egui::RichText::new(format!(
                                    "{} {}%{}",
                                    phase_label(pct), pct,
                                    if eta == usize::MAX {
                                        String::new()
                                    } else {
                                        format!(" — about {} left", fmt_eta(eta))
                                    },
                                )).size(11.0).color(txt_dim(ui.visuals().dark_mode)));
                                if ui.add(egui::Button::new("✖ Abort").small()).clicked() {
                                    self.analyzer.abort_preprocess();
                                    self.status_msg = "Analysis aborted.".into();
                                }
                            } else if !self.status_msg.is_empty() {
                                ui.label(egui::RichText::new(&self.status_msg)
                                    .size(11.0).color(txt_dim(ui.visuals().dark_mode)));
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Budget:");
                            let mut unlimited = self.analyzer.cache_budget_gb <= 0.0;
                            if ui.checkbox(&mut unlimited, "Unlimited")
                                .on_hover_text("No eviction — the cache grows until you clear it.")
                                .changed()
                            {
                                self.analyzer.cache_budget_gb =
                                    if unlimited { 0.0 } else { DEFAULT_CACHE_BUDGET_GB };
                            }
                            if !unlimited {
                                let mut gb = self.analyzer.cache_budget_gb;
                                // Logarithmic: the useful range runs from a
                                // couple of superlet tracks to a whole library,
                                // which linear steps cannot cover usefully.
                                let changed = ui.add(egui::Slider::new(&mut gb, 0.1..=500.0)
                                    .logarithmic(true)
                                    .suffix(" GB")
                                    .custom_formatter(|v, _| if v < 1.0 {
                                        format!("{:.0} MB", v * 1000.0)
                                    } else {
                                        format!("{v:.1}")
                                    }))
                                    .on_hover_text(
                                        "Ceiling on the whole cache directory. Least-recently-used \
                                         files are evicted after each analysis to stay under it. \
                                         Type a number here to set it exactly.\n\nA superlet track \
                                         at 180 fps costs 45–80 MB and cannot be packed much \
                                         smaller — the low byte of every stored value is \
                                         quantisation noise, so no lossless coder beats ~12 %. \
                                         Bounding the total is the real control.")
                                    .changed();
                                if changed { self.analyzer.cache_budget_gb = gb.max(0.05); }
                                // What that budget actually holds, in the units
                                // the user is spending it in.
                                let per_track = 60.0; // MB, typical superlet track
                                ui.label(egui::RichText::new(format!(
                                    "≈ {:.0} superlet tracks", gb * 1000.0 / per_track,
                                )).size(10.0).color(txt_faint(ui.visuals().dark_mode)));
                            }
                            let busy = self.analyzer.is_analyzing.load(Ordering::Relaxed);
                            if ui.add_enabled(
                                self.analyzer.cache_budget_gb > 0.0 && !busy,
                                egui::Button::new("Trim now").small(),
                            ).on_hover_text("Evict least-recently-used caches down to the budget.")
                                .clicked()
                            {
                                let budget = (self.analyzer.cache_budget_gb as f64 * 1e9) as u64;
                                let (n, freed) = evict_cache_to_budget(budget, None);
                                self.cache_stats_at = None;
                                self.status_msg = if n == 0 {
                                    "Cache already within budget.".into()
                                } else {
                                    format!("Removed {n} cache file(s), freed {:.1} MB",
                                            freed as f64 / 1e6)
                                };
                            }
                        });
                        let (count, bytes) = self.cache_stats;
                        let size_str = if bytes >= 1_000_000_000 {
                            format!("{:.1} GB", bytes as f64 / 1e9)
                        } else if bytes >= 1_000_000 {
                            format!("{:.1} MB", bytes as f64 / 1e6)
                        } else {
                            format!("{:.0} KB", bytes as f64 / 1e3)
                        };
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(
                                format!("Cache: {} file{} — {}", count, if count == 1 { "" } else { "s" }, size_str))
                                .size(10.0).color(txt_faint(ui.visuals().dark_mode)));
                            let analyzing = self.analyzer.is_analyzing.load(Ordering::Relaxed);
                            let over = self.analyzer.cache_budget_gb > 0.0
                                && bytes as f64 > self.analyzer.cache_budget_gb as f64 * 1e9;
                            if over {
                                ui.label(egui::RichText::new("over budget")
                                    .size(10.0).color(Color32::from_rgb(220, 160, 60)));
                            }
                            if ui.add_enabled(count > 0 && !analyzing,
                                egui::Button::new("🗑 Clear All").small()).clicked() {
                                let dir = home_dir().join(".moosik").join("cache");
                                if let Ok(entries) = std::fs::read_dir(&dir) {
                                    for e in entries.filter_map(|e| e.ok()) {
                                        let p = e.path();
                                        if p.extension().map(|x| x == "spectrumcache").unwrap_or(false) {
                                            let _ = std::fs::remove_file(p);
                                        }
                                    }
                                }
                                self.analyzer.clear_pre_frames();
                                self.needs_reanalysis = self.mode == SpectrumMode::PreProcess
                                    && self.current_path.is_some();
                                self.cache_stats_at = None;
                                self.status_msg = "All caches cleared.".into();
                            }
                        });
                    }); }
                }

                // ── Peak Hold settings (Bars only) ────────────────────────
                if self.style == VizStyle::Bars && self.show_peak_settings {
                    ui.group(|ui| {
                            ui.checkbox(&mut self.peak_config.enabled, "Enabled");
                            ui.add_enabled_ui(self.peak_config.enabled, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Hold time:");
                                    ui.add(egui::Slider::new(&mut self.peak_config.hold_ms, 10.0..=1000.0)
                                        .suffix(" ms"))
                                        .on_hover_text("How long the peak line freezes before it starts to decay.");
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Decay mode:");
                                    ui.selectable_value(&mut self.peak_config.decay_mode, PeakDecayMode::Linear,  "Linear")
                                        .on_hover_text("Peak falls at a constant speed.");
                                    ui.selectable_value(&mut self.peak_config.decay_mode, PeakDecayMode::Gravity, "Gravity")
                                        .on_hover_text("Peak accelerates as it falls — feels physical.");
                                    ui.selectable_value(&mut self.peak_config.decay_mode, PeakDecayMode::FadeOut, "Fade Out")
                                        .on_hover_text("Peak stays in place but fades to transparent.");
                                });
                                let speed_label = match self.peak_config.decay_mode {
                                    PeakDecayMode::FadeOut => "Fade speed:",
                                    _                      => "Fall speed:",
                                };
                                ui.horizontal(|ui| {
                                    ui.label(speed_label);
                                    ui.add(egui::Slider::new(&mut self.peak_config.fall_speed, 0.05..=5.0)
                                        .logarithmic(true))
                                        .on_hover_text("Initial decay rate (normalized units/sec). For Fade Out this controls how fast the line disappears.");
                                });
                                if self.peak_config.decay_mode == PeakDecayMode::Gravity {
                                    ui.horizontal(|ui| {
                                        ui.label("Acceleration:");
                                        ui.add(egui::Slider::new(&mut self.peak_config.acceleration, 0.5..=20.0)
                                            .logarithmic(true))
                                            .on_hover_text("How quickly the fall accelerates. Higher values make it feel heavier.");
                                    });
                                }
                                ui.horizontal(|ui| {
                                    ui.label("Thickness:");
                                    let mut t = self.peak_config.peak_thickness as i32;
                                    if ui.add(egui::Slider::new(&mut t, 1..=6).suffix(" px"))
                                        .on_hover_text("Height of the peak marker in physical pixels.")
                                        .changed()
                                    {
                                        self.peak_config.peak_thickness = t as u8;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Color:");
                                    let mut rgb = [
                                        self.peak_config.color.r(),
                                        self.peak_config.color.g(),
                                        self.peak_config.color.b(),
                                    ];
                                    if ui.color_edit_button_srgb(&mut rgb).changed() {
                                        self.peak_config.color = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                                    }
                                });
                            });
                        });
                }

                // ── Album Art settings ────────────────────────────────────
                if self.show_art_settings { ui.group(|ui| {
                        let has_art   = self.current_art.is_some();
                        let has_track = self.current_path.is_some();
                        // Disable spectrum art controls when a track is loaded with no art
                        let art_enabled = has_art || !has_track;
                        let track_key = self.current_path.as_ref()
                            .map(|p| p.to_string_lossy().into_owned())
                            .unwrap_or_default();

                        // ── Playlist thumbnails ──────────────────────────
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut self.art_settings.playlist_show, "Playlist thumbnails");
                            ui.add_space(6.0);
                            ui.add_enabled(
                                self.art_settings.playlist_show,
                                egui::Checkbox::new(
                                    &mut self.art_settings.playlist_placeholder,
                                    "Placeholder when no art",
                                ),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.checkbox(
                                &mut self.art_settings.spectrum_placeholder,
                                "Spectrum placeholder (no art)",
                            );
                        });

                        ui.separator();

                        // ── Spectrum scope: global vs per-track ──────────
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Spectrum:").size(11.0)
                                .color(txt_dim(ui.visuals().dark_mode)));
                            if !track_key.is_empty() {
                                let has_override = self.art_settings.has_override(&track_key);
                                let btn = if has_override { "This track ★" } else { "Global" };
                                let tip = if has_override {
                                    "Per-track override active. Click to revert to global."
                                } else {
                                    "Using global settings. Click to create a per-track override."
                                };
                                if ui.small_button(btn).on_hover_text(tip).clicked() {
                                    if has_override {
                                        self.art_settings.reset_to_global(&track_key);
                                    } else {
                                        self.art_settings.make_per_track(&track_key);
                                    }
                                }
                            }
                        });

                        // Resolve which config block to mutate
                        let use_per_track = !track_key.is_empty()
                            && self.art_settings.has_override(&track_key);
                        let cfg: &mut ArtDisplaySettings = if use_per_track {
                            self.art_settings.per_track.get_mut(&track_key).unwrap()
                        } else {
                            &mut self.art_settings.global
                        };

                        ui.add_enabled_ui(art_enabled, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Mode:");
                                ui.selectable_value(
                                    &mut cfg.spectrum_mode, ArtSpectrumMode::Hidden, "Hidden");
                                ui.selectable_value(
                                    &mut cfg.spectrum_mode, ArtSpectrumMode::Transparent, "Transparent")
                                    .on_hover_text("Art behind bars at reduced opacity");
                                ui.selectable_value(
                                    &mut cfg.spectrum_mode, ArtSpectrumMode::Mask, "Mask")
                                    .on_hover_text("Art visible only inside bar columns\n(Bars mode only)");
                            });

                            if cfg.spectrum_mode != ArtSpectrumMode::Hidden {
                                ui.horizontal(|ui| {
                                    ui.label("Fit:");
                                    ui.selectable_value(&mut cfg.fit, ArtFit::Preserve, "Preserve ratio");
                                    ui.selectable_value(&mut cfg.fit, ArtFit::Stretch, "Stretch to fill");
                                });
                            }

                            match cfg.spectrum_mode {
                                ArtSpectrumMode::Transparent => {
                                    ui.horizontal(|ui| {
                                        ui.label("Opacity:");
                                        ui.add(egui::Slider::new(&mut cfg.transparency, 0.0..=1.0)
                                            .step_by(0.01));
                                    });
                                }
                                ArtSpectrumMode::Mask => {
                                    ui.horizontal(|ui| {
                                        ui.label("Brightness:");
                                        ui.selectable_value(
                                            &mut cfg.mask_mode, ArtMaskMode::Dynamic, "Dynamic")
                                            .on_hover_text("Bar amplitude drives art brightness");
                                        ui.selectable_value(
                                            &mut cfg.mask_mode, ArtMaskMode::Fixed, "Fixed")
                                            .on_hover_text("Fixed brightness level, set by slider");
                                    });
                                    if cfg.mask_mode == ArtMaskMode::Fixed {
                                        ui.horizontal(|ui| {
                                            ui.label("Level:");
                                            ui.add(egui::Slider::new(&mut cfg.mask_brightness, 0.0..=1.0)
                                                .step_by(0.01));
                                        });
                                    }
                                }
                                ArtSpectrumMode::Hidden => {}
                            }
                        });

                        if has_track && !has_art {
                            ui.label(egui::RichText::new("ℹ No embedded art in this track.")
                                .size(10.0).color(txt_faint(ui.visuals().dark_mode)));
                        }
                    }); }

                // ── EQ panel ──────────────────────────────────────────────
                if self.show_eq {
                    egui::CollapsingHeader::new("🎛 Parametric EQ")
                        .default_open(true)
                        .show(ui, |ui| {
                            // EQ is bypassed whenever the session is on an
                            // output device stream. That is a routing fact,
                            // and it is what `bit_perfect` carries here — the
                            // app sets it from `on_bp_stream()`, not from the
                            // toggle.
                            //
                            // Whether the route is *exact* is a different
                            // question with its own answer, and drawing a
                            // green diamond for the first while claiming the
                            // second is the defect. This says only what it
                            // knows.
                            if self.bit_perfect {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(
                                            "\u{25C7} EQ is bypassed — this track plays on an \
                                             output device stream with no processing in the path",
                                        )
                                        .size(12.0)
                                        .color(txt_faint(ui.visuals().dark_mode)),
                                    );
                                });
                                ui.separator();
                            }

                            // Global controls row
                            ui.horizontal(|ui| {
                                let eq_on = { self.eq_state.lock().unwrap().enabled };
                                let on_label = egui::RichText::new(if eq_on { "ON" } else { "OFF" })
                                    .color(if eq_on { txt_ok(ui.visuals().dark_mode) } else { txt_faint(ui.visuals().dark_mode) });
                                if ui.selectable_label(eq_on, on_label)
                                    .on_hover_text("Bypass all EQ bands").clicked()
                                {
                                    let mut eq = self.eq_state.lock().unwrap();
                                    eq.enabled = !eq.enabled;
                                    eq.bump();
                                }
                                ui.separator();
                                ui.label("Overlay:");
                                ui.selectable_value(&mut self.eq_overlay, EqOverlayMode::Curve,       "Curve")
                                    .on_hover_text("Draw EQ response curve on top of spectrum bars");
                                ui.selectable_value(&mut self.eq_overlay, EqOverlayMode::ApplyToBars, "Apply")
                                    .on_hover_text("Apply EQ gain to bar heights — bars show the EQ'd spectrum");
                                ui.selectable_value(&mut self.eq_overlay, EqOverlayMode::Both,        "Both")
                                    .on_hover_text("Draw curve AND apply EQ gain to bars");
                            });

                            ui.separator();

                            // ── Preset bar ──────────────────────────────────
                            {
                                // Collect lists first (avoid borrow conflicts)
                                let global_presets: Vec<(u64, String, bool)> = self.preset_library.global
                                    .iter()
                                    .map(|p| (p.id, p.name.clone(), self.preset_library.default_id == Some(p.id)))
                                    .collect();
                                let local_presets: Vec<(u64, String)> = self.track_key()
                                    .map(|key| self.preset_library.local_presets(&key)
                                        .iter().map(|p| (p.id, p.name.clone())).collect())
                                    .unwrap_or_default();

                                // Refresh modified flag (happens every frame, cheap)
                                self.refresh_preset_modified();

                                // ── Pending-switch prompt (Save / Discard / Cancel) ──
                                if let Some(pending) = self.pending_preset_switch.clone() {
                                    ui.horizontal(|ui| {
                                        ui.label(egui::RichText::new("⚠ Unsaved changes — ").color(txt_warn(ui.visuals().dark_mode)));
                                        if ui.button("Save & switch").clicked() {
                                            self.overwrite_active_preset();
                                            let p = pending.clone();
                                            self.apply_preset_ref(p);
                                            self.pending_preset_switch = None;
                                        }
                                        if ui.button("Discard & switch").clicked() {
                                            let p = pending.clone();
                                            self.apply_preset_ref(p);
                                            self.pending_preset_switch = None;
                                        }
                                        if ui.button("Cancel").clicked() {
                                            self.pending_preset_switch = None;
                                        }
                                    });
                                    ui.separator();
                                }

                                // Helper: display name for active preset
                                let active_label = |lib: &EqPresetLibrary, active: &Option<PresetRef>, modified: bool| -> String {
                                    match active {
                                        None => if modified { "(custom) *".into() } else { "(none)".into() },
                                        Some(r) => {
                                            let key_str = lib.per_track.keys().next().map(|s| s.as_str()).unwrap_or("");
                                            let name = lib.find(r, Some(key_str))
                                                .map(|p| p.name.clone())
                                                .unwrap_or_else(|| "?".into());
                                            if modified { format!("{} *", name) } else { name }
                                        }
                                    }
                                };

                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("Preset:").size(11.0).color(txt_dim(ui.visuals().dark_mode)));

                                    // ── Global preset dropdown ──
                                    let global_label = {
                                        let is_global = self.active_preset.as_ref().map(|r| r.scope == PresetScope::Global).unwrap_or(false);
                                        if is_global {
                                            active_label(&self.preset_library, &self.active_preset, self.preset_modified)
                                        } else {
                                            "Global…".into()
                                        }
                                    };
                                    egui::ComboBox::from_id_salt("eq_preset_global")
                                        .selected_text(global_label)
                                        .width(130.0)
                                        .show_ui(ui, |ui| {
                                            for (id, name, is_default) in &global_presets {
                                                let display = if *is_default { format!("★ {}", name) } else { name.clone() };
                                                let is_active = self.active_preset.as_ref()
                                                    .map(|r| r.scope == PresetScope::Global && r.id == *id)
                                                    .unwrap_or(false);
                                                let r = PresetRef { scope: PresetScope::Global, id: *id };
                                                if ui.selectable_label(is_active, display).clicked() {
                                                    if self.preset_modified {
                                                        self.pending_preset_switch = Some(r);
                                                    } else {
                                                        self.apply_preset_ref(r);
                                                    }
                                                }
                                            }
                                        });

                                    // ── Local (song) preset dropdown ──
                                    if !local_presets.is_empty() || self.current_path.is_some() {
                                        let local_label = {
                                            let is_local = self.active_preset.as_ref().map(|r| r.scope == PresetScope::Local).unwrap_or(false);
                                            if is_local {
                                                active_label(&self.preset_library, &self.active_preset, self.preset_modified)
                                            } else {
                                                "Song…".into()
                                            }
                                        };
                                        egui::ComboBox::from_id_salt("eq_preset_local")
                                            .selected_text(local_label)
                                            .width(120.0)
                                            .show_ui(ui, |ui| {
                                                for (id, name) in &local_presets {
                                                    let is_active = self.active_preset.as_ref()
                                                        .map(|r| r.scope == PresetScope::Local && r.id == *id)
                                                        .unwrap_or(false);
                                                    let r = PresetRef { scope: PresetScope::Local, id: *id };
                                                    if ui.selectable_label(is_active, name).clicked() {
                                                        if self.preset_modified {
                                                            self.pending_preset_switch = Some(r);
                                                        } else {
                                                            self.apply_preset_ref(r);
                                                        }
                                                    }
                                                }
                                            });
                                    }

                                    // ── Modified action buttons ──
                                    if self.preset_modified && self.pending_preset_switch.is_none() {
                                        if self.active_preset.is_some()
                                            && ui.small_button("Update").on_hover_text("Overwrite active preset with current bands").clicked() {
                                            self.overwrite_active_preset();
                                        }
                                        if ui.small_button("Discard").on_hover_text("Revert bands to saved preset").clicked() {
                                            if let Some(r) = self.active_preset.clone() {
                                                self.apply_preset_ref(r);
                                            } else {
                                                let mut eq = self.eq_state.lock().unwrap();
                                                eq.bands.clear(); eq.bump();
                                                self.preset_modified = false;
                                            }
                                        }
                                    }

                                    // ── Save As New toggle ──
                                    let save_new_label = if self.eq_save_new_open { "▾ Save As New" } else { "▸ Save As New" };
                                    if ui.small_button(save_new_label).clicked() {
                                        if !self.eq_save_new_open {
                                            // Pre-fill a sensible name
                                            let track_key = self.track_key();
                                            self.eq_save_name_buf = self.preset_library.auto_name(&self.eq_save_new_scope, track_key.as_deref());
                                        }
                                        self.eq_save_new_open = !self.eq_save_new_open;
                                    }
                                });

                                // ── Save As New expanded row ──
                                if self.eq_save_new_open {
                                    ui.horizontal(|ui| {
                                        ui.label(egui::RichText::new("Name:").size(11.0));
                                        ui.add(egui::TextEdit::singleline(&mut self.eq_save_name_buf).desired_width(140.0));
                                        ui.label(egui::RichText::new("Scope:").size(11.0));
                                        ui.selectable_value(&mut self.eq_save_new_scope, PresetScope::Global, "Global");
                                        if self.current_path.is_some() {
                                            ui.selectable_value(&mut self.eq_save_new_scope, PresetScope::Local, "Song");
                                        }
                                        let name_ok = !self.eq_save_name_buf.trim().is_empty();
                                        if ui.add_enabled(name_ok, egui::Button::new("💾 Save")).clicked() {
                                            let name = self.eq_save_name_buf.trim().to_string();
                                            let scope = self.eq_save_new_scope.clone();
                                            self.save_as_new_preset(name, scope);
                                            self.eq_save_new_open = false;
                                            self.eq_save_name_buf.clear();
                                        }
                                        if ui.small_button("✕").clicked() {
                                            self.eq_save_new_open = false;
                                        }
                                    });
                                }

                                // ── Per-preset actions (Rename, Duplicate, Delete, Set Default) ──
                                if let Some(active_ref) = self.active_preset.clone() {
                                    // Rename state
                                    if let Some((ref rename_ref, ref mut rename_buf)) = self.eq_rename_state.clone()
                                        && *rename_ref == active_ref {
                                        ui.horizontal(|ui| {
                                                ui.label(egui::RichText::new("Rename:").size(11.0));
                                                // We need mutable access — pull from field
                                                if let Some((_, ref mut buf)) = self.eq_rename_state {
                                                    ui.add(egui::TextEdit::singleline(buf).desired_width(140.0));
                                                }
                                                let new_name = rename_buf.trim().to_string();
                                                let ok = !new_name.is_empty();
                                                if ui.add_enabled(ok, egui::Button::new("✓")).clicked() {
                                                    let key = self.track_key();
                                                    if let Some(p) = self.preset_library.find_mut(&active_ref, key.as_deref()) { p.name = new_name; }
                                                    self.preset_library.save();
                                                    self.eq_rename_state = None;
                                                }
                                                if ui.small_button("✕").clicked() {
                                                    self.eq_rename_state = None;
                                                }
                                            });
                                    }

                                    ui.horizontal(|ui| {
                                        // Rename button
                                        if (self.eq_rename_state.is_none() || self.eq_rename_state.as_ref().map(|(r,_)| r) != Some(&active_ref))
                                            && ui.small_button("✏ Rename").clicked() {
                                            let key = self.track_key();
                                            let cur_name = self.preset_library.find(&active_ref, key.as_deref())
                                                .map(|p| p.name.clone()).unwrap_or_default();
                                            self.eq_rename_state = Some((active_ref.clone(), cur_name));
                                        }

                                        // Duplicate
                                        if ui.small_button("⧉ Duplicate").on_hover_text("Save a copy as a new preset").clicked() {
                                            let key = self.track_key();
                                            let (cur_name, cur_bands) = self.preset_library.find(&active_ref, key.as_deref())
                                                .map(|p| (format!("{} copy", p.name), p.bands.clone()))
                                                .unwrap_or_else(|| (self.preset_library.auto_name(&active_ref.scope, key.as_deref()), self.eq_state.lock().unwrap().bands.clone()));
                                            let id = self.preset_library.alloc_id();
                                            let new_preset = EqPreset { id, name: cur_name, bands: cur_bands };
                                            let scope = active_ref.scope.clone();
                                            match &scope {
                                                PresetScope::Global => self.preset_library.global.push(new_preset),
                                                PresetScope::Local => {
                                                    let k = key.unwrap_or_default();
                                                    self.preset_library.per_track.entry(k).or_default().push(new_preset);
                                                }
                                            }
                                            self.preset_library.save();
                                            self.active_preset = Some(PresetRef { scope, id });
                                            self.preset_modified = false;
                                        }

                                        // Set as Default (global only)
                                        if active_ref.scope == PresetScope::Global {
                                            let is_default = self.preset_library.default_id == Some(active_ref.id);
                                            let default_label = if is_default { "★ Default" } else { "☆ Set Default" };
                                            if ui.small_button(default_label).on_hover_text("Use this preset for tracks that have no last-used preset").clicked() {
                                                if is_default {
                                                    self.preset_library.default_id = None;
                                                } else {
                                                    self.preset_library.default_id = Some(active_ref.id);
                                                }
                                                self.preset_library.save();
                                            }
                                        }

                                        // Delete / confirm-delete
                                        if self.eq_confirm_delete.as_ref() == Some(&active_ref) {
                                            ui.label(egui::RichText::new("Delete?").color(Color32::from_rgb(240, 80, 80)));
                                            if ui.small_button("Yes").clicked() {
                                                let key = self.track_key();
                                                self.preset_library.delete(&active_ref, key.as_deref());
                                                self.preset_library.save();
                                                self.active_preset = None;
                                                self.preset_modified = !self.eq_state.lock().unwrap().bands.is_empty();
                                                self.eq_confirm_delete = None;
                                            }
                                            if ui.small_button("No").clicked() {
                                                self.eq_confirm_delete = None;
                                            }
                                        } else if ui.small_button("🗑 Delete").clicked() {
                                            self.eq_confirm_delete = Some(active_ref.clone());
                                        }
                                    });
                                }
                            }

                            ui.separator();

                            // Band table
                            let bands_snap: Vec<EqBand> = self.eq_state.lock().unwrap().bands.clone();
                            let sr = self.eq_state.lock().unwrap().sample_rate;

                            egui::Grid::new("eq_band_grid")
                                .num_columns(8)
                                .striped(true)
                                .spacing([4.0, 2.0])
                                .show(ui, |ui| {
                                    ui.label(egui::RichText::new("#").size(10.0).color(txt_faint(ui.visuals().dark_mode)));
                                    ui.label(egui::RichText::new("Type").size(10.0).color(txt_faint(ui.visuals().dark_mode)));
                                    ui.label(egui::RichText::new("Freq (Hz)").size(10.0).color(txt_faint(ui.visuals().dark_mode)));
                                    ui.label(egui::RichText::new("Gain (dB)").size(10.0).color(txt_faint(ui.visuals().dark_mode)));
                                    ui.label(egui::RichText::new("Q").size(10.0).color(txt_faint(ui.visuals().dark_mode)));
                                    ui.label(egui::RichText::new("On").size(10.0).color(txt_faint(ui.visuals().dark_mode)));
                                    ui.label(egui::RichText::new("Del").size(10.0).color(txt_faint(ui.visuals().dark_mode)));
                                    ui.end_row();

                                    let mut remove_idx: Option<usize> = None;
                                    for (i, _) in bands_snap.iter().enumerate() {
                                        let col = eq_band_color(i);
                                        ui.label(egui::RichText::new(format!("{}", i + 1)).color(col).size(11.0));

                                        // Band kind selector
                                        let mut eq = self.eq_state.lock().unwrap();
                                        let kind = eq.bands[i].kind.clone();
                                        egui::ComboBox::from_id_salt(format!("eq_kind_{}", i))
                                            .selected_text(kind.label())
                                            .width(58.0)
                                            .show_ui(ui, |ui| {
                                                let mut changed = false;
                                                for k in [BandKind::Peaking, BandKind::LowShelf, BandKind::HighShelf, BandKind::HighPass, BandKind::LowPass, BandKind::Notch] {
                                                    let sel = ui.selectable_value(&mut eq.bands[i].kind, k.clone(), k.label()).changed();
                                                    if sel { changed = true; }
                                                }
                                                if changed { eq.bump(); }
                                            });

                                        // Freq
                                        let freq_changed = ui.add(egui::DragValue::new(&mut eq.bands[i].freq)
                                            .range(20.0..=20_000.0).speed(1.0).suffix(" Hz")).changed();
                                        if freq_changed { eq.bump(); }

                                        // Gain (only for band types that have gain)
                                        if eq.bands[i].kind.has_gain() {
                                            let g_changed = ui.add(egui::DragValue::new(&mut eq.bands[i].gain_db)
                                                .range(-EQ_DB_RANGE..=EQ_DB_RANGE).speed(0.1).suffix(" dB")).changed();
                                            if g_changed { eq.bump(); }
                                        } else {
                                            ui.label("—");
                                        }

                                        // Q
                                        let q_changed = ui.add(egui::DragValue::new(&mut eq.bands[i].q)
                                            .range(0.1..=10.0).speed(0.01)).changed();
                                        if q_changed { eq.bump(); }

                                        // Enable toggle
                                        let en_changed = ui.checkbox(&mut eq.bands[i].enabled, "").changed();
                                        if en_changed { eq.bump(); }

                                        drop(eq);

                                        // Delete
                                        if ui.small_button("✕").clicked() {
                                            remove_idx = Some(i);
                                        }

                                        ui.end_row();
                                    }

                                    if let Some(idx) = remove_idx {
                                        let mut eq = self.eq_state.lock().unwrap();
                                        if idx < eq.bands.len() { eq.bands.remove(idx); eq.bump(); }
                                    }
                                });

                            ui.horizontal(|ui| {
                                if ui.button("➕ Add band").on_hover_text("Add a new peaking band at 1 kHz.\nYou can also click on the spectrum to add a band there.").clicked() {
                                    let mut eq = self.eq_state.lock().unwrap();
                                    if eq.bands.len() < 16 {
                                        eq.bands.push(EqBand::default_peak(1000.0));
                                        eq.bump();
                                    }
                                }
                                let band_count = self.eq_state.lock().unwrap().bands.len();
                                if band_count > 0 && ui.button("🗑 Clear all").clicked() {
                                    let mut eq = self.eq_state.lock().unwrap();
                                    eq.bands.clear();
                                    eq.bump();
                                }
                                ui.label(egui::RichText::new(format!("{}/16 bands", band_count))
                                    .size(10.0).color(txt_faint(ui.visuals().dark_mode)));
                            });

                            let _ = sr; // used in overlay, suppress warning
                        });
                }

                ui.separator();

                let avail = ui.available_size();
                let sense = if self.show_eq { egui::Sense::click_and_drag() } else { egui::Sense::hover() };
                let (rect, spec_response) = ui.allocate_exact_size(avail, sense);
                if ui.is_rect_visible(rect) {
                    let painter = ui.painter_at(rect);
                    painter.rect_filled(rect, 2.0, Color32::from_rgb(8, 10, 16));
                    // Inset plot area to make room for axis labels
                    let plot_rect = Rect::from_min_max(
                        Pos2::new(rect.left() + DB_MARGIN, rect.top()),
                        Pos2::new(rect.right(), rect.bottom() - FREQ_MARGIN),
                    );

                    // ── EQ bar modification (apply EQ gain to displayed bars) ──
                    // Skipped in bit-perfect mode: EQ is not in the audio path, so
                    // the spectrum correctly shows what you hear without modification.
                    let eq_snap = if self.show_eq && !self.bit_perfect {
                        let eq = self.eq_state.lock().unwrap();
                        if eq.is_active() { Some((eq.bands.clone(), eq.sample_rate, eq.enabled)) } else { None }
                    } else { None };

                    let owned_mags: Vec<f32>;
                    let mags: &[f32] = if let Some((ref bands, sr, _)) = eq_snap {
                        if matches!(self.eq_overlay, EqOverlayMode::ApplyToBars | EqOverlayMode::Both) {
                            owned_mags = self.analyzer.magnitudes.iter().enumerate().map(|(i, &m)| {
                                let freq = bar_center_freq(i, self.analyzer.magnitudes.len(), self.min_freq, self.max_freq);
                                let gain = 10f32.powf(total_eq_response_db(bands, sr, freq) / 20.0);
                                (m * gain).clamp(0.0, 1.0)
                            }).collect();
                            &owned_mags
                        } else {
                            &self.analyzer.magnitudes
                        }
                    } else {
                        &self.analyzer.magnitudes
                    };

                    // ── Album art background / placeholder ───────────────
                    let track_key = self.current_path.as_ref()
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    let art_cfg = self.art_settings.settings_for(&track_key).clone();
                    let has_art = self.current_art.is_some();

                    if has_art {
                        if let Some((tex_id, art_w, art_h)) = self.current_art {
                            match &art_cfg.spectrum_mode {
                                ArtSpectrumMode::Transparent => {
                                    let alpha = (art_cfg.transparency * 255.0) as u8;
                                    let art_rect = art_dest_rect(plot_rect, art_w, art_h, &art_cfg.fit);
                                    painter.image(
                                        tex_id, art_rect,
                                        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                                        Color32::from_white_alpha(alpha),
                                    );
                                }
                                ArtSpectrumMode::Hidden | ArtSpectrumMode::Mask => {}
                            }
                        }
                    } else if self.art_settings.spectrum_placeholder {
                        painter.rect_filled(plot_rect, 4.0, Color32::from_gray(14));
                        painter.text(
                            plot_rect.center(),
                            egui::Align2::CENTER_CENTER,
                            "♪",
                            egui::FontId::proportional(56.0),
                            Color32::from_gray(28),
                        );
                    }

                    let pal = Palette::new(self.palette_kind, self.palette_accent);

                    // Channel views take over the plot when they are both
                    // selected and possible. `effective_view` returns Mix
                    // whenever they are not, and `draw_channel_view` returns
                    // false if the frame it was handed lacks a channel it
                    // needs — so a missing channel falls back rather than
                    // drawing something that is not it.
                    // The corner readouts share one strip and have to be
                    // measured before any of them is drawn, because the legend
                    // sits between the other two and its start depends on how
                    // wide the LUFS number happens to be.
                    // Read before the row is planned: the badge is part of it,
                    // and it is only drawn when the plot is not already saying
                    // the same thing in the middle. A pure read of analyser
                    // state, so moving it earlier changes nothing else.
                    let plot_is_silent = self.analyzer.channel_frame().is_silent();
                    let analysis_text = (self
                        .analyzer
                        .is_analyzing
                        .load(Ordering::Relaxed)
                        && !plot_is_silent)
                        .then(|| {
                            let pct =
                                self.analyzer.analysis_progress.load(Ordering::Relaxed);
                            let eta = self.analyzer.eta_secs.load(Ordering::Relaxed);
                            if eta == usize::MAX {
                                format!("analysing {pct}%")
                            } else {
                                format!("analysing {pct}%  ~{}", fmt_eta(eta))
                            }
                        });
                    let lufs_text = (self.style != VizStyle::Phasescope).then(|| {
                        if self.momentary_lufs.is_finite() {
                            format!("{:.1} LUFS", self.momentary_lufs)
                        } else {
                            "— LUFS".to_string()
                        }
                    });
                    let fps_text = format!("{:.0} fps", self.current_fps);
                    let top_row = TopRow::plan(
                        &painter,
                        plot_rect,
                        lufs_text.as_deref(),
                        &fps_text,
                        analysis_text.as_deref(),
                    );

                    let ch_view = channels::effective_view(
                        self.channel_view, &self.channel_availability,
                    );
                    let drew_channels = ch_view != channels::ChannelView::Mix && {
                        let frame = self.analyzer.channel_frame();
                        self.draw_channel_view(&painter, ch_view, &frame, plot_rect, &pal, top_row)
                    };
                    let drew_diff = uses_diff_axes(ch_view, drew_channels);

                    match self.style {
                        _ if drew_channels => {}
                        VizStyle::Bars => {
                            if has_art && matches!(art_cfg.spectrum_mode, ArtSpectrumMode::Mask) {
                                if let Some((tex_id, art_w, art_h)) = self.current_art {
                                    draw_bars_art_mask(
                                        &painter, mags, plot_rect, self.bar_gap,
                                        tex_id, art_w, art_h, &art_cfg.fit,
                                        &art_cfg.mask_mode, art_cfg.mask_brightness,
                                    );
                                } else {
                                    draw_bars(&painter, mags, plot_rect, self.bar_gap, &pal);
                                }
                            } else {
                                draw_bars(&painter, mags, plot_rect, self.bar_gap, &pal);
                            }
                            if self.peak_config.enabled && !self.peak_vals.is_empty() {
                                draw_peak_hold(
                                    &painter, &self.peak_vals, &self.peak_alphas,
                                    plot_rect, self.bar_gap, &self.peak_config,
                                );
                            }
                        }
                        VizStyle::Line       => draw_line(&painter, mags, plot_rect, pal.line()),
                        VizStyle::FilledArea => draw_filled(&painter, mags, plot_rect, &pal),
                        VizStyle::Waterfall  => {
                            self.update_waterfall_texture(vp_ctx, &pal);
                            self.draw_waterfall(&painter, plot_rect);
                        },
                        VizStyle::Spectrogram => {
                            if self.spectrogram.dirty {
                                let img = self.spectrogram.to_color_image();
                                if let Some(ref mut th) = self.spectrogram_texture {
                                    th.set(img, egui::TextureOptions::LINEAR);
                                } else {
                                    let th = vp_ctx.load_texture(
                                        "moosik_spectrogram", img,
                                        egui::TextureOptions::LINEAR,
                                    );
                                    self.spectrogram_texture = Some(th);
                                }
                                self.spectrogram.dirty = false;
                            }
                            if let Some(ref th) = self.spectrogram_texture {
                                painter.image(
                                    th.id(), plot_rect,
                                    egui::Rect::from_min_max(
                                        egui::Pos2::ZERO,
                                        egui::Pos2::new(1.0, 1.0),
                                    ),
                                    Color32::WHITE,
                                );
                            }
                        }
                        VizStyle::OctaveBands => {
                            draw_octave_bands(
                                &painter, &self.octave_bands, plot_rect,
                                self.analyzer.sample_rate, self.min_freq, self.max_freq, &pal,
                            );
                        }
                        VizStyle::Phasescope => {
                            // Use the snapshot captured in tick() — no lock or clone here
                            draw_phasescope(&painter, &self.phasescope_frames, plot_rect, self.correlation);
                        }
                    }
                    // ── Analysis-status overlay ───────────────────────────
                    // Shown only when the plot is genuinely dead. DSD is the
                    // main case: playback is a DoP carrier or raw native DSD,
                    // not analysable PCM, so there
                    // is no live signal to fall back on and the first analysis
                    // of a track would otherwise be an unexplained blank panel.
                    if plot_is_silent && !matches!(self.style, VizStyle::Phasescope) {
                        let analyzing = self.analyzer.is_analyzing.load(Ordering::Relaxed);
                        let is_dsd = self.current_path.as_deref().is_some_and(crate::dsd::is_dsd_path);
                        if self.mode == SpectrumMode::PreProcess
                            && analyzing && self.analyzer.pre_frames.is_empty()
                        {
                            let pct = self.analyzer.analysis_progress.load(Ordering::Relaxed);
                            let label = if is_dsd {
                                format!("Analyzing DSD…  {pct}%")
                            } else {
                                format!("Analyzing…  {pct}%")
                            };
                            let center = plot_rect.center();
                            painter.text(
                                Pos2::new(center.x, center.y - 14.0),
                                egui::Align2::CENTER_CENTER,
                                label,
                                egui::FontId::proportional(15.0),
                                Color32::from_gray(200),
                            );
                            // Slim progress bar under the text.
                            let bar_w = 220.0_f32.min(plot_rect.width() * 0.6);
                            let bar = Rect::from_center_size(
                                Pos2::new(center.x, center.y + 10.0),
                                egui::Vec2::new(bar_w, 6.0),
                            );
                            painter.rect_filled(bar, 3.0, Color32::from_gray(30));
                            let fill_w = bar_w * (pct.min(100) as f32 / 100.0);
                            if fill_w > 1.0 {
                                let fill = Rect::from_min_size(bar.min, egui::Vec2::new(fill_w, 6.0));
                                painter.rect_filled(fill, 3.0, pal.line());
                            }
                            // Superlet analyses run for minutes, not seconds, so
                            // the wait needs a number attached to it and a way
                            // out. Both are shown for every mode — an ETA is
                            // just as welcome on a fast one.
                            let eta = self.analyzer.eta_secs.load(Ordering::Relaxed);
                            painter.text(
                                Pos2::new(center.x, center.y + 26.0),
                                egui::Align2::CENTER_CENTER,
                                if eta == usize::MAX {
                                    "estimating…".to_string()
                                } else {
                                    format!("about {} remaining", fmt_eta(eta))
                                },
                                egui::FontId::proportional(11.0),
                                Color32::from_gray(140),
                            );
                            let abort_rect = Rect::from_center_size(
                                Pos2::new(center.x, center.y + 48.0),
                                egui::Vec2::new(76.0, 22.0),
                            );
                            if ui.put(abort_rect, egui::Button::new(
                                egui::RichText::new("Abort").size(11.0)
                            )).clicked() {
                                self.analyzer.abort_preprocess();
                            }
                        } else if is_dsd && self.mode == SpectrumMode::RealTime {
                            painter.text(
                                plot_rect.center(),
                                egui::Align2::CENTER_CENTER,
                                "Real-time FFT isn't available for DSD (playback is DoP, not PCM) — switch to Pre-Process mode",
                                egui::FontId::proportional(13.0),
                                Color32::from_gray(150),
                            );
                        }
                    }

                    // Axis labels. The Diff view has its own pair, because
                    // neither of its axes is what the ordinary ones describe —
                    // see `draw_diff_axes`. This keys off `drew_diff`, not off
                    // the *requested* view, so a Diff selection that fell back
                    // to Mix gets the ordinary axes it is actually showing.
                    if drew_diff {
                        draw_diff_axes(
                            &painter, plot_rect, self.diff_layout,
                            self.analyzer.sample_rate, self.min_freq, self.max_freq,
                            self.analyzer.aslt_cfg.scale,
                        );
                    } else {
                        // Skip for spectrogram (its own freq axis is baked in),
                        // octave bands (draws its own labels below bars), and phasescope.
                        if !matches!(self.style, VizStyle::Spectrogram | VizStyle::OctaveBands | VizStyle::Phasescope) {
                            draw_db_labels(&painter, plot_rect);
                        }
                        if !matches!(self.style, VizStyle::Spectrogram | VizStyle::Phasescope) {
                            draw_freq_labels(
                                &painter, plot_rect, self.analyzer.sample_rate,
                                self.min_freq, self.max_freq,
                                self.analyzer.aslt_cfg.scale,
                            );
                        }
                    }
                    // The EQ overlay is a curve in gain-against-frequency drawn
                    // on the plot's own coordinates, and its draggable nodes are
                    // hit-tested in them. Diff has neither of those axes, so the
                    // curve would be meaningless and every node would sit at the
                    // wrong frequency and gain. Rotating the interaction to suit
                    // is a larger piece of work than this change; until then the
                    // overlay stands down and says so. Nothing about the EQ
                    // itself changes — the bands, their gains and the EQ panel
                    // are all untouched.
                    if self.show_eq && drew_diff {
                        painter.text(
                            Pos2::new(plot_rect.left() + 6.0, plot_rect.bottom() - 4.0),
                            egui::Align2::LEFT_BOTTOM,
                            "EQ overlay off in Diff — this plot is not dB against frequency. \
                             EQ settings are unchanged; edit them in the EQ panel.",
                            egui::FontId::proportional(10.0),
                            Color32::from_gray(105),
                        );
                    }
                    // ── EQ overlay and node interaction ──────────────────────
                    if self.show_eq && !drew_diff && !matches!(self.style, VizStyle::Phasescope | VizStyle::Waterfall | VizStyle::Spectrogram) {
                        let (bands, sr) = {
                            let eq = self.eq_state.lock().unwrap();
                            (eq.bands.clone(), eq.sample_rate)
                        };
                        let draw_curve = matches!(self.eq_overlay, EqOverlayMode::Curve | EqOverlayMode::Both);
                        draw_eq_overlay(&painter, &bands, sr, plot_rect,
                            self.min_freq, self.max_freq, draw_curve,
                            self.eq_hovered_node, self.eq_dragging_node,
                            self.analyzer.aslt_cfg.scale);

                        // Mouse hit-test: find nearest node within 14px

                        if let Some(ptr) = spec_response.hover_pos() {
                            self.eq_hovered_node = None;
                            for (i, band) in bands.iter().enumerate() {
                                if !band.enabled { continue; }
                                let np = eq_node_pos(band, plot_rect, self.min_freq, self.max_freq,
                                                     self.analyzer.aslt_cfg.scale);
                                if (np - ptr).length() < 14.0 {
                                    self.eq_hovered_node = Some(i);
                                    break;
                                }
                            }
                        }

                        if spec_response.drag_started() {
                            self.eq_dragging_node = self.eq_hovered_node;
                        }
                        if spec_response.drag_stopped() {
                            self.eq_dragging_node = None;
                        }

                        if let Some(drag_idx) = self.eq_dragging_node {
                            let delta = spec_response.drag_delta();
                            if delta != egui::Vec2::ZERO {
                                let mut eq = self.eq_state.lock().unwrap();
                                if let Some(band) = eq.bands.get_mut(drag_idx) {
                                    // Horizontal → frequency (log scale)
                                    // Along whichever axis is on screen, so the
                                    // node tracks the pointer instead of
                                    // sliding away from it.
                                    let fscale = self.analyzer.aslt_cfg.scale;
                                    let t = fscale.position_of(
                                        band.freq, self.min_freq, self.max_freq,
                                    ) + delta.x / plot_rect.width();
                                    band.freq = fscale
                                        .freq_at(t.clamp(0.0, 1.0), self.min_freq, self.max_freq)
                                        .clamp(20.0, 20_000.0);
                                    // Vertical → gain
                                    if band.kind.has_gain() {
                                        let db_per_px = EQ_DB_RANGE / (plot_rect.height() * 0.5);
                                        band.gain_db = (band.gain_db - delta.y * db_per_px).clamp(-EQ_DB_RANGE, EQ_DB_RANGE);
                                    }
                                    eq.bump();
                                }
                            }
                        }

                        // Right-click on node → remove
                        if spec_response.secondary_clicked()
                            && let Some(idx) = self.eq_hovered_node {
                            let mut eq = self.eq_state.lock().unwrap();
                            if idx < eq.bands.len() { eq.bands.remove(idx); eq.bump(); }
                            self.eq_hovered_node = None;
                        }

                        // Left-click on empty space → add band (up to 16)
                        if spec_response.clicked() && self.eq_hovered_node.is_none() && self.eq_dragging_node.is_none()
                            && let Some(click) = spec_response.interact_pointer_pos()
                            && plot_rect.contains(click) {
                            let t = (click.x - plot_rect.left()) / plot_rect.width();
                            let freq = self.analyzer.aslt_cfg.scale
                                .freq_at(t, self.min_freq, self.max_freq)
                                .clamp(20.0, 20_000.0);
                            let db_per_px = EQ_DB_RANGE / (plot_rect.height() * 0.5);
                            let gain = ((plot_rect.center().y - click.y) * db_per_px).clamp(-EQ_DB_RANGE, EQ_DB_RANGE);
                            let mut eq = self.eq_state.lock().unwrap();
                            if eq.bands.len() < 16 {
                                let mut band = EqBand::default_peak(freq);
                                band.gain_db = gain;
                                eq.bands.push(band);
                                eq.bump();
                            }
                        }
                    }

                    // The two fixed readouts, in the strip planned above and
                    // drawn last so they sit over the bars. Both are inside the
                    // plot rather than in the dB margin, which a Diff plot uses
                    // for its own axis labels.
                    painter.text(
                        Pos2::new(top_row.fps_x, top_row.y),
                        egui::Align2::RIGHT_TOP,
                        &fps_text,
                        readout_font(),
                        Color32::from_rgba_unmultiplied(180, 180, 180, 120),
                    );
                    if let Some(lufs) = &lufs_text {
                        painter.text(
                            Pos2::new(top_row.lufs_x, top_row.y),
                            egui::Align2::LEFT_TOP,
                            lufs,
                            readout_font(),
                            Color32::from_rgba_unmultiplied(180, 220, 180, 160),
                        );
                    }
                    // Analysis progress, wherever the plot happens to be.
                    //
                    // The centred "Analyzing..." text below only appears when
                    // there is nothing else to draw, so re-analysing a track
                    // that already has frames on screen used to show no
                    // progress at all unless the Cache panel was open — which
                    // is not somewhere anyone would think to look for it.
                    // Anywhere the centred notice is not already saying it.
                    //
                    // This used to require `!pre_frames.is_empty()`, to pair with
                    // the centred notice's `pre_frames.is_empty()`. That looked
                    // like a clean split and was not: a *first* analysis clears
                    // pre_frames, and the live-FFT fallback keeps drawing bars,
                    // so the plot is not silent either — the centred notice was
                    // ruled out by the plot and this one by the frames, and the
                    // most important case in the app showed no progress at all.
                    // The real condition is the plot, which is what the centred
                    // notice actually keys on.
                    if let Some(text) = &analysis_text {
                        let dark = ui.visuals().dark_mode;
                        let pct = self.analyzer.analysis_progress.load(Ordering::Relaxed);
                        // The same string the row was planned around, so the
                        // space reserved is the space used. Re-reading the
                        // atomics here would let the text grow past its slot
                        // between the two reads and land on the frame rate
                        // again, which is the bug this is fixing.
                        let pos = Pos2::new(top_row.analysis_x, top_row.y + 2.0);
                        let gal = painter.layout_no_wrap(
                            text.clone(), badge_font(), txt_accent(dark),
                        );
                        let bg = egui::Rect::from_min_size(
                            Pos2::new(pos.x - gal.size().x - BADGE_PAD, pos.y - 3.0),
                            gal.size() + egui::vec2(2.0 * BADGE_PAD, 6.0),
                        );
                        painter.rect_filled(
                            bg, 3.0, Color32::from_rgba_unmultiplied(0, 0, 0, 120));
                        // A bar under the text, so the rate is readable at a
                        // glance without reading the number.
                        let track = egui::Rect::from_min_size(
                            Pos2::new(bg.left() + 4.0, bg.bottom() + 2.0),
                            egui::vec2(bg.width() - 8.0, 2.0),
                        );
                        painter.rect_filled(
                            track, 1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 40));
                        let done = egui::Rect::from_min_size(
                            track.min,
                            egui::vec2(track.width() * (pct as f32 / 100.0).clamp(0.0, 1.0), 2.0),
                        );
                        painter.rect_filled(done, 1.0, txt_accent(dark));
                        painter.galley(
                            Pos2::new(bg.left() + BADGE_PAD, bg.top() + 3.0),
                            gal,
                            txt_accent(dark),
                        );
                    }

                    // Debug overlay (F3)
                    if self.show_debug {
                        let interp_name = format!("{:?}", self.interp_mode);
                        let mapping_name = format!("{:?}", self.bar_mapping);
                        let mode_name = format!("{:?}", self.mode);
                        let analyzing = self.analyzer.is_analyzing.load(Ordering::Relaxed);
                        let pct = self.analyzer.analysis_progress.load(Ordering::Relaxed);
                        let mut lines = vec![
                            format!("── Spectrum Debug ──────────────────"),
                            format!("Mode:        {}", mode_name),
                            format!("FFT size:    {}  pad: {}×  padded: {}", self.fft_size, self.pad_factor, self.fft_size * self.pad_factor),
                            format!("Overlap:     {:.0}%  hop: {} smp ({:.1} fps)", self.overlap * 100.0,
                                self.analyzer.pre_hop(),
                                self.analyzer.sample_rate as f32 / self.analyzer.pre_hop() as f32),
                            format!("Interp:      {}  mapping: {}", interp_name, mapping_name),
                            format!("Bars:        {}  sr: {} Hz", self.bar_count, self.analyzer.sample_rate),
                            format!("freq range:  {:.0}–{:.0} Hz", self.min_freq, self.max_freq),
                            format!("── Pre-process ─────────────────────"),
                            format!("Analyzing:   {}  progress: {}%", analyzing, pct),
                            format!("Frames:      {}  frame_rate: {:.2} fps", self.analyzer.pre_frames.len(), self.analyzer.pre_frame_rate),
                            format!("── Runtime ─────────────────────────"),
                            format!("Tick rate:   {:.1}/s (max {:.0}) — analyser, not frames",
                                self.current_fps, self.max_fps),
                            format!("Src rows:    {:.1}/s consumed from cache",
                                self.pre_rows_per_sec),
                            format!("Channels:    {} — {:.1} extra FFT/s",
                                self.channel_view.label(), self.channel_ffts_per_sec),
                            format!("REPAINT:     {:.1} fps (real)", self.real_repaint_fps),
                            format!("DRAW cost:   {:.2} ms/frame", self.draw_ms),
                            format!("LUFS:        {:.2}", self.momentary_lufs),
                            format!("Correlation: {:.3}", self.correlation),
                            format!("FFT norms:   {} bins", self.analyzer.last_fft_norms.len()),
                        ];

                        // Where the last pre-process actually spent itself.
                        //
                        // A GPU monitor reads the device at a fraction of its
                        // capacity during an analysis, and the obvious reading
                        // — "the shader is too small" — is wrong. The phases
                        // inside a group run one after another, so the device
                        // idles through everything that is not `device`. This
                        // panel shows that directly, because four rewrites of
                        // this path were aimed at the wrong phase for want of
                        // exactly these six numbers.
                        let st = aslt::stats();
                        lines.push("── Pre-process cost ────────────────".into());
                        if st.total_s <= 0.0 {
                            lines.push("(no analysis has run this session)".into());
                        } else {
                            lines.push(format!("Wall clock:  {:.2} s", st.total_s));
                            lines.push(format!(
                                "Route:       {}",
                                match aslt::gpu_device_name() {
                                    Some(n) if st.used_gpu() => n,
                                    Some(n) => format!("{n} (idle — no batch qualified)"),
                                    None => "CPU only (no device)".into(),
                                },
                            ));
                            if st.used_gpu() {
                                lines.push(format!(
                                    "Bars:        {} on device, {} on cores",
                                    st.gpu_bars, st.cpu_bars,
                                ));
                                lines.push(format!(
                                    "Batches:     {}  kernels: {}", st.chunks, st.kernels,
                                ));
                                // Thread-summed, so these exceed wall clock on a
                                // parallel run; they are for weighing the phases
                                // against each other, which is the whole point.
                                lines.push(format!(
                                    "  build      {:>7.2} s   (kernel taps)", st.build_s,
                                ));
                                lines.push(format!(
                                    "  stage      {:>7.2} s   (signal upload)", st.stage_s,
                                ));
                                lines.push(format!(
                                    "  device     {:>7.2} s   (GPU)", st.device_s,
                                ));
                                lines.push(format!(
                                    "  fill       {:>7.2} s   (edges + CPU wavelets)", st.fill_s,
                                ));
                                lines.push(format!(
                                    "Device duty: {:.0}% of the whole analysis",
                                    st.device_duty() * 100.0,
                                ));
                            }
                            // Which block sizes this machine has decided on.
                            // "(probe)" means the synthetic bootstrap is still
                            // in charge and real A/B samples are still coming.
                            lines.push("── Calibration (this machine) ──────".into());
                            lines.extend(gpu_calib::describe());
                        }
                        let x = rect.left() + 8.0;
                        let mut y = rect.top() + 20.0;
                        let line_h = 13.0;
                        for line in &lines {
                            painter.text(
                                Pos2::new(x + 1.0, y + 1.0),
                                egui::Align2::LEFT_TOP, line,
                                egui::FontId::monospace(11.0),
                                Color32::from_rgba_unmultiplied(0, 0, 0, 180),
                            );
                            painter.text(
                                Pos2::new(x, y),
                                egui::Align2::LEFT_TOP, line,
                                egui::FontId::monospace(11.0),
                                Color32::from_rgba_unmultiplied(255, 220, 80, 220),
                            );
                            y += line_h;
                        }
                    }

                }
            });
        });

        // Persist any view-setting changes made this frame (throttled).
        self.persist_settings_if_changed();

        // Record the wall-clock cost of this full draw (EMA-smoothed).
        let ms = show_start.elapsed().as_secs_f32() * 1000.0;
        self.draw_ms = self.draw_ms * 0.9 + ms * 0.1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Source-time smoothing, through the production tick
// ---------------------------------------------------------------------------

/// These drive `SpectrumAnalyzer::tick_pre` itself rather than re-deciding the
/// same questions in test-local helpers. The pure arithmetic already has its
/// own tests in `timing`; what is checked here is that the analyser is actually
/// wired to it — that a row is consumed exactly once, that the result does not
/// depend on how often the UI asks, and that the discontinuity paths clear the
/// cursor.
#[cfg(test)]
mod source_time_smoothing_tests {
    use super::*;

    const RATE: f64 = 180.0;
    const BARS: usize = 8;

    /// Every mapping shares one consumption path, so each of them has to show
    /// the same temporal behaviour. Mapping selects how the *cache* is built,
    /// which these tests take as given; what varies here is only that nothing
    /// in the tick reads it and branches.
    const MAPPINGS: [BarMappingMode; 4] = [
        BarMappingMode::FlatOverlap,
        BarMappingMode::Gaussian,
        BarMappingMode::Cqt,
        BarMappingMode::Superlet,
    ];

    /// Distinguishable rows: bar `b` of row `f` is a function of both, so a
    /// swapped, skipped or repeated row shows up as a wrong number rather than
    /// coincidentally matching.
    fn cache(n_frames: usize) -> Vec<Vec<f32>> {
        (0..n_frames)
            .map(|f| {
                (0..BARS)
                    .map(|b| (((f * 7 + b * 13) % 97) as f32) / 97.0)
                    .collect()
            })
            .collect()
    }

    fn analyzer(frames: Vec<Vec<f32>>, smoothing: f32, mapping: &BarMappingMode)
        -> SpectrumAnalyzer
    {
        let n = frames[0].len();
        let mut a = SpectrumAnalyzer::new(new_sample_buf());
        a.bar_count = n;
        a.magnitudes = vec![0.0; n];
        a.smoothed = vec![0.0; n];
        a.peak_input = vec![0.0; n];
        // Pre-process reads its own retention; `smoothing` is Real-time's.
        a.pre_smoothing = smoothing;
        a.bar_mapping = mapping.clone();
        a.waterfall_enabled = true;
        a.set_pre_frames(frames, RATE);
        a
    }

    /// Wall time per tick is irrelevant to the pre-process path, which filters
    /// in source time; a plausible value is passed so the argument is not
    /// quietly ignored by being zero everywhere.
    fn tick(a: &mut SpectrumAnalyzer, at: f64, ui_hz: f64) {
        a.tick_pre(at, 1.0 / ui_hz);
    }

    /// Smoothing at zero has to mean off. It did not: this path hard-coded
    /// `old * 0.5 + new * 0.5` whatever the slider said, so "off" still halved
    /// every step toward the new value and the setting was inert.
    #[test]
    fn zero_smoothing_shows_the_raw_cached_row() {
        for mapping in &MAPPINGS {
            let frames = cache(64);
            let mut a = analyzer(frames.clone(), 0.0, mapping);
            for f in 0..40 {
                tick(&mut a, f as f64 / RATE, 60.0);
            }
            assert_eq!(
                a.magnitudes, frames[39],
                "{mapping:?}: smoothing 0 must display the cache row itself"
            );
        }
    }

    /// The same, with equal-loudness weighting on: "raw" means the weighted
    /// row, since the weighting is a display correction rather than filtering.
    #[test]
    fn zero_smoothing_shows_the_eq_adjusted_row() {
        let frames = cache(32);
        let mut a = analyzer(frames.clone(), 0.0, &BarMappingMode::Cqt);
        a.eq_weights = (0..BARS).map(|b| b as f32 - 3.5).collect();
        for f in 0..20 {
            tick(&mut a, f as f64 / RATE, 60.0);
        }
        let want: Vec<f32> = frames[19]
            .iter()
            .enumerate()
            .map(|(b, &v)| (v + (b as f32 - 3.5) / 80.0).clamp(0.0, 1.0))
            .collect();
        assert_eq!(a.magnitudes, want);
    }

    /// A repeated tick inside one source row must change nothing at all. The
    /// old code ran another EMA step per repaint, which is why the visible
    /// decay tracked the monitor.
    #[test]
    fn repeating_a_tick_within_one_row_is_idempotent() {
        for mapping in &MAPPINGS {
            let mut a = analyzer(cache(64), 0.75, mapping);
            for f in 0..10 {
                tick(&mut a, f as f64 / RATE, 60.0);
            }
            let mags = a.magnitudes.clone();
            let seq = a.waterfall_seq;
            let rows = a.waterfall.len();
            // Six more ticks inside the same row, as a 1080 Hz repaint would.
            for k in 0..6 {
                tick(&mut a, 9.0 / RATE + k as f64 * 1e-5, 1080.0);
            }
            assert_eq!(a.magnitudes, mags, "{mapping:?}: magnitudes moved");
            assert_eq!(a.waterfall_seq, seq, "{mapping:?}: waterfall advanced");
            assert_eq!(a.waterfall.len(), rows, "{mapping:?}: waterfall grew");
        }
    }

    /// The property the whole change exists for: at one source timestamp the
    /// display is the same whatever rate the UI ran at to get there.
    #[test]
    fn the_result_does_not_depend_on_the_ui_rate() {
        for mapping in &MAPPINGS {
            let mut results = Vec::new();
            for ui_hz in [60.0f64, 144.0, 180.0, 240.0] {
                let mut a = analyzer(cache(512), 0.75, mapping);
                let ticks = ui_hz as usize;
                for k in 0..=ticks {
                    // Lands exactly on t = 1.0 s for every rate.
                    tick(&mut a, k as f64 / ui_hz, ui_hz);
                }
                results.push((ui_hz, a.magnitudes.clone(), a.waterfall_seq));
            }
            let (_, ref first, first_seq) = results[0];
            for (hz, mags, seq) in &results {
                assert_eq!(
                    mags, first,
                    "{mapping:?}: {hz} Hz UI gave a different spectrum at t=1s"
                );
                assert_eq!(
                    seq, &first_seq,
                    "{mapping:?}: {hz} Hz UI produced a different waterfall length"
                );
            }
        }
    }

    /// Crossing several rows in one tick must equal having ticked once per row.
    /// On a 60 Hz display against the 180 fps analysis default the UI crosses
    /// three rows a tick, and the old
    /// code discarded two of them.
    #[test]
    fn a_multi_row_tick_equals_the_rows_consumed_one_at_a_time() {
        for mapping in &MAPPINGS {
            let mut coarse = analyzer(cache(256), 0.75, mapping);
            let mut fine = analyzer(cache(256), 0.75, mapping);
            // 60 Hz UI against a 180 fps cache: three rows per tick.
            for k in 0..=30 {
                tick(&mut coarse, k as f64 / 60.0, 60.0);
            }
            for f in 0..=90 {
                tick(&mut fine, f as f64 / RATE, RATE);
            }
            assert_eq!(coarse.last_pre_frame, fine.last_pre_frame);
            assert_eq!(
                coarse.magnitudes, fine.magnitudes,
                "{mapping:?}: a 3-row tick differs from three 1-row ticks"
            );
        }
    }

    /// The same impulse, followed all the way to the values the renderer
    /// draws.
    ///
    /// `peak_input` is an input. Asserting it proves the analyser found the
    /// transient, not that the marker shows it — `update_peaks` is the consumer
    /// and it lives on the window, so a change that stopped it reading the
    /// interval maximum would leave the previous test green.
    #[test]
    fn a_skipped_impulse_reaches_the_marker_the_renderer_draws() {
        let mut w = SpectrumWindow::new();
        w.mode = SpectrumMode::PreProcess;
        w.style = VizStyle::Bars;
        w.peak_config.enabled = true;
        w.bar_count = BARS;
        w.analyzer.bar_count = BARS;
        w.analyzer.pre_smoothing = 0.0;
        w.analyzer.magnitudes = vec![0.0; BARS];
        w.analyzer.smoothed = vec![0.0; BARS];
        w.analyzer.peak_input = vec![0.0; BARS];

        let mut frames = vec![vec![0.1f32; BARS]; 64];
        frames[4] = vec![0.9f32; BARS];
        w.analyzer.set_pre_frames(frames, RATE);

        // A 60 Hz display against a 180 Hz cache lands on rows 3 and 6, never 4.
        w.last_fft_time = None;
        w.tick(3.0 / RATE, true);
        w.last_fft_time = None;
        w.tick(6.0 / RATE, true);

        assert_eq!(
            w.analyzer.magnitudes[0], 0.1,
            "the displayed bar should be the latest row, not the peak"
        );
        assert!(
            w.peak_vals[0] > 0.85,
            "the peak marker the renderer draws is {}, not the impulse",
            w.peak_vals[0]
        );
        assert!(w.peak_alphas[0] > 0.0, "the marker is drawn fully transparent");
    }

    /// A transient one row wide, in a row the UI never lands on, still has to
    /// reach the peak marker.
    #[test]
    fn a_skipped_impulse_survives_in_the_interval_peak() {
        for mapping in &MAPPINGS {
            let mut frames = vec![vec![0.1f32; BARS]; 64];
            // Row 4 is crossed by a 60 Hz tick that lands on rows 3 and 6.
            frames[4] = vec![0.9f32; BARS];
            let mut a = analyzer(frames, 0.0, mapping);
            tick(&mut a, 3.0 / RATE, 60.0);
            tick(&mut a, 6.0 / RATE, 60.0);
            assert_eq!(
                a.magnitudes[0], 0.1,
                "{mapping:?}: the displayed value is the latest row, not the peak"
            );
            assert_eq!(
                a.peak_input[0], 0.9,
                "{mapping:?}: the impulse in row 4 was lost"
            );
        }
    }

    /// Backwards movement is a discontinuity, not a cascade. A looping track
    /// wraps to row 0 and a seek can land anywhere.
    #[test]
    fn a_backward_jump_snaps_and_resets_the_filter() {
        let frames = cache(256);
        let mut a = analyzer(frames.clone(), 0.9, &BarMappingMode::Cqt);
        for k in 0..=30 {
            tick(&mut a, k as f64 / 60.0, 60.0);
        }
        assert!(a.magnitudes != frames[0], "setup: filter should be far from row 0");
        // Loop wrap.
        tick(&mut a, 0.0, 60.0);
        assert_eq!(a.last_pre_frame, Some(0));
        assert_eq!(
            a.magnitudes, frames[0],
            "a backward jump must snap, not smear from the old position"
        );
    }

    /// A seek uses the same snap, and clears the interval peak with it.
    #[test]
    fn snap_pre_to_adopts_the_target_row_exactly() {
        let frames = cache(512);
        let mut a = analyzer(frames.clone(), 0.9, &BarMappingMode::Cqt);
        for k in 0..=30 {
            tick(&mut a, k as f64 / 60.0, 60.0);
        }
        a.invalidate_pre_cursor();
        a.snap_pre_to(400.0 / RATE);
        assert_eq!(a.last_pre_frame, Some(400));
        assert_eq!(a.magnitudes, frames[400]);
        assert_eq!(a.peak_input, frames[400]);
        // And the next ordinary tick continues from there rather than snapping.
        tick(&mut a, 401.0 / RATE, 60.0);
        assert_eq!(a.last_pre_frame, Some(401));
    }

    /// Replacing the cache invalidates the cursor: row 40 of the old analysis
    /// says nothing about row 40 of the new one.
    #[test]
    fn replacing_the_cache_resets_the_cursor() {
        let mut a = analyzer(cache(256), 0.75, &BarMappingMode::Cqt);
        for k in 0..=20 {
            tick(&mut a, k as f64 / 60.0, 60.0);
        }
        assert!(a.last_pre_frame.is_some());
        let replacement = cache(256);
        a.set_pre_frames(replacement.clone(), RATE);
        assert_eq!(a.last_pre_frame, None, "set_pre_frames must drop the cursor");
        // The next tick snaps rather than cascading across the join.
        a.smoothing = 0.9;
        tick(&mut a, 60.0 / RATE, 60.0);
        assert_eq!(a.magnitudes, replacement[60]);
    }

    /// Track change and stop both go through `reset`.
    #[test]
    fn reset_and_bar_count_changes_drop_the_cursor() {
        let mut a = analyzer(cache(256), 0.75, &BarMappingMode::Cqt);
        for k in 0..=20 {
            tick(&mut a, k as f64 / 60.0, 60.0);
        }
        assert!(a.last_pre_frame.is_some());
        a.reset();
        assert_eq!(a.last_pre_frame, None);

        let mut b = analyzer(cache(256), 0.75, &BarMappingMode::Cqt);
        for k in 0..=20 {
            tick(&mut b, k as f64 / 60.0, 60.0);
        }
        b.set_bar_count(BARS * 2);
        assert_eq!(b.last_pre_frame, None);
    }

    /// A long seek must cost a bounded number of row operations, not one per
    /// row crossed. Counted through the production counter rather than inferred.
    #[test]
    fn a_large_jump_does_a_bounded_amount_of_work() {
        let mut a = analyzer(cache(120_000), 0.75, &BarMappingMode::Cqt);
        tick(&mut a, 0.0, 60.0);
        let _ = a.take_rows_consumed();
        // Ten minutes forward at 180 fps: 108 000 rows crossed.
        tick(&mut a, 600.0, 60.0);
        let consumed = a.take_rows_consumed();
        assert_eq!(consumed, 1, "a jump past the limit must snap, not replay");
        assert_eq!(a.last_pre_frame, Some(108_000));

        // And the largest legitimate catch-up stays on the replay path.
        let mut b = analyzer(cache(4096), 0.75, &BarMappingMode::Cqt);
        tick(&mut b, 0.0, 60.0);
        let _ = b.take_rows_consumed();
        let n = timing::MAX_CATCHUP_FRAMES as f64;
        tick(&mut b, n / RATE, 60.0);
        assert_eq!(b.take_rows_consumed(), timing::MAX_CATCHUP_FRAMES as u64);
    }

    /// The waterfall advances on source time — one row per cached row consumed
    /// — so its axis follows the analysis and not the display.
    #[test]
    fn the_waterfall_receives_one_row_per_consumed_source_row() {
        for ui_hz in [60.0f64, 144.0, 240.0] {
            let mut a = analyzer(cache(1024), 0.5, &BarMappingMode::Cqt);
            a.waterfall_rows = 4096;
            let seq0 = a.waterfall_seq;
            let ticks = ui_hz as usize;
            for k in 0..=ticks {
                tick(&mut a, k as f64 / ui_hz, ui_hz);
            }
            let rows = a.waterfall_seq - seq0;
            // One second of source at RATE rows/s, plus the initial snap. The
            // display rate does not enter into it: whatever the UI schedule,
            // the same cached rows are crossed and every one produces a row.
            let want = RATE as u64;
            assert!(
                rows.abs_diff(want) <= 2,
                "{ui_hz} Hz UI produced {rows} rows for {want} consumed source rows"
            );
        }
    }

    /// Smoothing is presentation-only and must stay out of the cache identity,
    /// or every nudge of the slider would invalidate an analysis that takes
    /// minutes to rebuild.
    #[test]
    fn smoothing_is_not_part_of_the_cache_key() {
        let path = PathBuf::from("C:/music/track.flac");
        let cfg = aslt::AsltPreset::Standard.config();
        let key = |_s: f32| {
            cache_path_for(
                &path,
                1024,
                8192,
                16,
                0.875,
                &WindowFn::Hann,
                20.0,
                24_000.0,
                &BarMappingMode::Superlet,
                &InterpolationMode::None,
                176_400,
                &cfg,
                180.0,
            )
        };
        // `cache_path_for` takes no smoothing argument at all; this asserts that
        // remains true, and that two different display settings therefore reach
        // the same analysis.
        assert_eq!(key(0.0), key(0.97));
    }
}

// ---------------------------------------------------------------------------
// Real-time left/right spectra
// ---------------------------------------------------------------------------

/// The interesting property here is not that two spectra can be produced; it is
/// that they are the *right* two, and that nothing produces a pair when there
/// is not one to produce. These drive the analyser and the tap themselves, not
/// a test-local copy of the availability table.
#[cfg(test)]
mod channel_spectrum_tests {
    use super::*;
    use channels::{availability, ChannelAvailability, ChannelView, NO_LIVE_TAP};

    const SR: u32 = 48_000;
    const FFT: usize = 1024;
    const BARS: usize = 256;

    fn analyzer() -> SpectrumAnalyzer {
        let mut a = SpectrumAnalyzer::new(new_sample_buf());
        a.sample_rate = SR;
        a.fft_size = FFT;
        a.bar_count = BARS;
        a.smoothing = 0.0;
        a.min_freq = 20.0;
        a.max_freq = 20_000.0;
        a.rebuild_fft();
        a.magnitudes = vec![0.0; BARS];
        a.smoothed = vec![0.0; BARS];
        a.peak_input = vec![0.0; BARS];
        a
    }

    fn tone(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / SR as f32).sin() * 0.5)
            .collect()
    }

    fn silence(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    /// Bar index whose centre frequency is nearest `freq`.
    fn bar_of(freq: f32) -> usize {
        let scale = freq_scale::FreqScale::default();
        (0..BARS)
            .min_by(|&a, &b| {
                let fa = scale.bar_center(a, BARS, 20.0, 20_000.0);
                let fb = scale.bar_center(b, BARS, 20.0, 20_000.0);
                (fa - freq).abs().partial_cmp(&(fb - freq).abs()).unwrap()
            })
            .unwrap()
    }

    /// Peak of a spectrum, and where it is.
    fn peak(bars: &[f32]) -> (usize, f32) {
        bars.iter()
            .enumerate()
            .fold((0usize, 0.0f32), |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            })
    }

    /// A tone in the left channel only must appear in Left and not in Right.
    #[test]
    fn a_left_only_tone_appears_only_on_the_left() {
        let mut a = analyzer();
        a.process_channels(&tone(1000.0, FFT), &silence(FFT), 1.0 / 60.0);
        let (li, lv) = peak(&a.bars_left);
        let (_, rv) = peak(&a.bars_right);
        assert!(
            (li as isize - bar_of(1000.0) as isize).abs() <= 2,
            "left peak at bar {li}, expected near {}",
            bar_of(1000.0)
        );
        assert!(lv > 0.5, "left peak too weak: {lv}");
        assert!(rv < 0.05, "a silent right channel showed {rv}");
    }

    #[test]
    fn a_right_only_tone_appears_only_on_the_right() {
        let mut a = analyzer();
        a.process_channels(&silence(FFT), &tone(1000.0, FFT), 1.0 / 60.0);
        let (_, lv) = peak(&a.bars_left);
        let (ri, rv) = peak(&a.bars_right);
        assert!((ri as isize - bar_of(1000.0) as isize).abs() <= 2);
        assert!(rv > 0.5, "right peak too weak: {rv}");
        assert!(lv < 0.05, "a silent left channel showed {lv}");
    }

    /// The channels must not leak into one another: two different tones stay
    /// where they were put. A swapped or averaged pair fails this.
    #[test]
    fn different_tones_stay_in_their_own_channels() {
        let mut a = analyzer();
        a.process_channels(&tone(500.0, FFT), &tone(5000.0, FFT), 1.0 / 60.0);
        let (li, _) = peak(&a.bars_left);
        let (ri, _) = peak(&a.bars_right);
        assert!(
            (li as isize - bar_of(500.0) as isize).abs() <= 2,
            "left peaked at {li}, wanted {}",
            bar_of(500.0)
        );
        assert!(
            (ri as isize - bar_of(5000.0) as isize).abs() <= 2,
            "right peaked at {ri}, wanted {}",
            bar_of(5000.0)
        );
        // And the 5 kHz energy is not visible on the left, nor 500 Hz on the
        // right — which is what a swap or a shared buffer would produce.
        assert!(a.bars_left[bar_of(5000.0)] < 0.05);
        assert!(a.bars_right[bar_of(500.0)] < 0.05);
    }

    /// Identical channels must read the same as the existing mono path.
    ///
    /// The tolerance is for the two routes reaching the same number by
    /// different arithmetic — the mono tap averages the pair before the window
    /// where this averages nothing — not for a difference in what is measured.
    #[test]
    fn equal_channels_match_the_mono_spectrum() {
        let sig = tone(1000.0, FFT);
        let mut a = analyzer();
        {
            let mut buf = a.sample_buf.lock().unwrap();
            buf.extend_from_slice(&sig);
        }
        a.process_realtime(1.0 / 60.0);
        let mono = a.magnitudes.clone();

        a.process_channels(&sig, &sig, 1.0 / 60.0);
        let frame = a.channel_frame();
        let mix = frame.mix;
        assert_eq!(mix.len(), mono.len());
        let (l, r) = frame.pair().expect("both channels present");

        for i in 0..BARS {
            assert!(
                (l[i] - mono[i]).abs() < 1e-5,
                "bar {i}: left {} vs mono {}",
                l[i],
                mono[i]
            );
            assert!((r[i] - mono[i]).abs() < 1e-5, "bar {i}: right vs mono");
        }
    }

    /// Selecting Mix must not cost a second and third transform.
    #[test]
    fn mix_runs_no_channel_transform() {
        let mut a = analyzer();
        {
            let mut buf = a.sample_buf.lock().unwrap();
            buf.extend_from_slice(&tone(1000.0, FFT));
        }
        let _ = a.take_channel_ffts();
        for _ in 0..10 {
            a.process_realtime(1.0 / 60.0);
        }
        assert_eq!(
            a.take_channel_ffts(),
            0,
            "the mono path must not run a channel FFT"
        );
        assert!(a.channel_frame().left.is_none());
        assert!(a.channel_frame().right.is_none());

        // And when a channel view does ask, exactly two run per frame.
        a.process_channels(&tone(500.0, FFT), &tone(500.0, FFT), 1.0 / 60.0);
        assert_eq!(a.take_channel_ffts(), 2);
    }

    /// Too little history is not a reason to keep showing the last frame.
    #[test]
    fn a_short_buffer_clears_rather_than_freezing() {
        let mut a = analyzer();
        a.process_channels(&tone(1000.0, FFT), &tone(1000.0, FFT), 1.0 / 60.0);
        assert!(a.channel_frame().pair().is_some());
        a.process_channels(&tone(1000.0, FFT / 4), &tone(1000.0, FFT / 4), 1.0 / 60.0);
        assert!(
            a.channel_frame().pair().is_none(),
            "a starved frame must clear, not repeat the previous one"
        );
    }

    /// Every state that cannot supply channels says which one it is, and the
    /// frame it yields carries no channels at all — so there is no shape in
    /// which the renderer could draw the mix twice.
    #[test]
    fn unavailable_states_yield_no_channels() {
        let cases: [(bool, bool, u16, ChannelAvailability); 5] = [
            (false, true, 1, ChannelAvailability::Mono),
            (false, true, 6, ChannelAvailability::Multichannel(6)),
            (false, true, NO_LIVE_TAP, ChannelAvailability::NoLiveTap),
            (true, true, 2, ChannelAvailability::PreProcessMono),
            (false, false, 2, ChannelAvailability::UnsupportedStyle),
        ];
        for (pre, style_ok, ch, want) in cases {
            let got = availability(pre, style_ok, ch);
            assert_eq!(got, want);
            assert!(!got.is_available());
            assert!(got.reason().is_some());
            assert_eq!(channels::effective_view(ChannelView::Split, &got), ChannelView::Mix);

            // The analyser is what the renderer reads, and with no channel
            // work done it offers none.
            let a = analyzer();
            let frame = a.channel_frame();
            assert!(frame.left.is_none() && frame.right.is_none());
            assert!(frame.pair().is_none());
        }
    }

    /// Resetting is what stops a restarted stream decaying from the previous
    /// track. Track change, seek, FFT size and mapping all route through it.
    #[test]
    fn reset_clears_the_channel_spectra() {
        let mut a = analyzer();
        a.process_channels(&tone(1000.0, FFT), &tone(1000.0, FFT), 1.0 / 60.0);
        assert!(a.channel_frame().pair().is_some());
        a.reset_channels();
        assert!(a.channel_frame().pair().is_none());

        // A different FFT size means different bins per bar, so rebuild_fft
        // drops the filter state with the plan.
        a.process_channels(&tone(1000.0, FFT), &tone(1000.0, FFT), 1.0 / 60.0);
        assert!(a.channel_frame().pair().is_some());
        a.rebuild_fft();
        assert!(a.channel_frame().pair().is_none());

        // And a full reset, which is what a track change reaches.
        a.process_channels(&tone(1000.0, FFT), &tone(1000.0, FFT), 1.0 / 60.0);
        a.reset();
        assert!(a.channel_frame().pair().is_none());
    }
}

/// The tap is the only thing that can say what the live stream is, so its
/// pairing and its lifecycle are checked directly rather than through the
/// analyser.
#[cfg(test)]
mod stereo_tap_tests {
    use super::*;
    use rodio::Source;

    /// A bare source that emits a known interleaved pattern.
    pub(super) struct Pattern {
        pub data: Vec<f32>,
        pub at: usize,
        pub channels: u16,
    }

    impl Iterator for Pattern {
        type Item = f32;
        fn next(&mut self) -> Option<f32> {
            let v = *self.data.get(self.at)?;
            self.at += 1;
            Some(v)
        }
    }

    impl Source for Pattern {
        fn current_frame_len(&self) -> Option<usize> { None }
        fn channels(&self) -> u16 { self.channels }
        fn sample_rate(&self) -> u32 { 48_000 }
        fn total_duration(&self) -> Option<Duration> { None }
    }

    /// Left samples are positive, right negative, so a swap is unmissable.
    pub(super) fn interleaved(frames: usize) -> Vec<f32> {
        (0..frames)
            .flat_map(|i| [1.0 + i as f32, -(1.0 + i as f32)])
            .collect()
    }

    /// The shared PCM tap must preserve `[L, R]` order, not merely collect
    /// pairs.
    #[test]
    fn the_shared_pcm_tap_preserves_left_right_order() {
        const FRAMES: usize = 4096;
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let src = Pattern { data: interleaved(FRAMES), at: 0, channels: 2 };
        let mut tapped = SpectrumSource::new(src, Arc::clone(&mono), Arc::clone(&stereo));
        while tapped.next().is_some() {}

        let guard = stereo.lock().unwrap();
        assert_eq!(guard.channels, 2, "the tap must publish the decoder count");
        assert!(!guard.frames.is_empty());
        for (i, &[l, r]) in guard.frames.iter().enumerate() {
            assert!(l > 0.0, "frame {i}: left {l} should be positive");
            assert!(r < 0.0, "frame {i}: right {r} should be negative");
            assert!((l + r).abs() < 1e-6, "frame {i}: {l} and {r} are not a pair");
        }
    }

    /// Mono material must publish one channel, and push no pairs at all.
    #[test]
    fn a_mono_source_publishes_one_channel_and_no_pairs() {
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let src = Pattern { data: vec![0.25; 2048], at: 0, channels: 1 };
        let mut tapped = SpectrumSource::new(src, Arc::clone(&mono), Arc::clone(&stereo));
        while tapped.next().is_some() {}

        let guard = stereo.lock().unwrap();
        assert_eq!(guard.channels, 1);
        assert!(
            guard.frames.is_empty(),
            "a mono source must not fabricate stereo frames"
        );
        assert_eq!(
            channels::availability(false, true, guard.channels),
            channels::ChannelAvailability::Mono
        );
    }

    /// The case native DSD lands in: nothing attaches a tap, so the buffer must
    /// declare that rather than leave the previous track in place.
    #[test]
    fn ending_a_stream_retracts_the_channel_claim() {
        let stereo = new_stereo_buf();
        let mono = new_sample_buf();
        let gen_a = begin_stereo_stream(&stereo, 2);
        {
            let mut g = stereo.lock().unwrap();
            g.frames.push([0.5, -0.5]);
        }
        assert_eq!(
            channels::availability(false, true, stereo.lock().unwrap().channels),
            channels::ChannelAvailability::Available
        );

        invalidate_live_pcm_route(&stereo, &mono);
        let g = stereo.lock().unwrap();
        assert_eq!(g.channels, channels::NO_LIVE_TAP);
        assert!(g.frames.is_empty(), "stale frames must not outlive the stream");
        assert_ne!(g.generation, gen_a, "the generation must move on");
        assert_eq!(
            channels::availability(false, true, g.channels),
            channels::ChannelAvailability::NoLiveTap
        );
    }

    /// A source still draining after the buffer has been claimed by something
    /// else must not write into it.
    ///
    /// It has to have claimed first, which now means being pulled: a source
    /// that was merely constructed holds no lease and has nothing to lose.
    #[test]
    fn a_superseded_tap_cannot_write() {
        const FRAMES: usize = 8192;
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let src = Pattern { data: interleaved(FRAMES), at: 0, channels: 2 };
        let mut old = SpectrumSource::new(src, Arc::clone(&mono), Arc::clone(&stereo));

        // Pull enough to claim and flush at least once.
        for _ in 0..(BATCH_SIZE * 4) {
            old.next();
        }
        assert!(
            !stereo.lock().unwrap().frames.is_empty(),
            "setup: the source should own the buffer by now"
        );

        // Something else takes the display mid-stream.
        let new_gen = begin_stereo_stream(&stereo, 2);
        for _ in 0..(BATCH_SIZE * 4) {
            old.next();
        }

        let g = stereo.lock().unwrap();
        assert_eq!(g.generation, new_gen, "the old source re-claimed the buffer");
        assert!(
            g.frames.is_empty(),
            "the superseded tap wrote into the new stream"
        );
    }

    /// The defect this whole ownership model exists for.
    ///
    /// `MoosikApp` restarts the route — which builds and starts the tap — and
    /// only *then* calls `on_play` with the new track's title and sample rate.
    /// `on_play` used to retire the lease, which bumped the generation past the
    /// tap that had just been created, so every write it made for the rest of
    /// the track was rejected and Left/Right never worked on ordinary playback.
    ///
    /// UI metadata must never revoke a route that has already started.
    #[test]
    fn on_play_does_not_revoke_a_route_that_has_already_started() {
        const FRAMES: usize = 8192;
        let mut w = SpectrumWindow::new();
        let mono = Arc::clone(&w.sample_buf);
        let stereo = Arc::clone(&w.stereo_buf);

        // 1. The engine starts the route and the mixer begins pulling.
        let src = Pattern { data: interleaved(FRAMES), at: 0, channels: 2 };
        let mut tap = SpectrumSource::new(src, mono, Arc::clone(&stereo));
        for _ in 0..(BATCH_SIZE * 4) {
            tap.next();
        }
        let live = stereo.lock().unwrap().generation;
        assert_eq!(stereo.lock().unwrap().channels, 2);

        // 2. Only now does the app tell the spectrum window what is playing.
        w.on_play(Path::new("C:/music/track.flac"), 48_000);

        assert_eq!(
            stereo.lock().unwrap().generation, live,
            "on_play moved the generation out from under the running tap"
        );
        assert_eq!(
            stereo.lock().unwrap().channels, 2,
            "on_play retracted the channel claim of a live route"
        );

        // 3. And the tap keeps feeding it — with frames written *after*
        //    `on_play`, not merely with the buffer still being non-empty. The
        //    earlier assertion accepted `len >= before`, which a tap that had
        //    stopped writing entirely satisfies.
        {
            let mut g = stereo.lock().unwrap();
            g.frames.clear();
        }
        for _ in 0..(BATCH_SIZE * 4) {
            tap.next();
        }
        let g = stereo.lock().unwrap();
        assert!(
            !g.frames.is_empty(),
            "the tap wrote nothing after on_play"
        );
        for (i, &[l, r]) in g.frames.iter().enumerate() {
            assert!(l > 0.0 && r < 0.0, "frame {i}: {l}/{r} is not a live pair");
        }
        assert_eq!(
            channels::availability(false, true, g.channels),
            channels::ChannelAvailability::Available,
            "Left/Right must still be offered on ordinary playback"
        );
    }

    /// Shared gapless appends the next track up to two seconds early. Until the
    /// mixer actually pulls from it, the track still playing owns the display.
    #[test]
    fn a_queued_source_does_not_take_the_display_until_it_is_pulled() {
        const FRAMES: usize = 8192;
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();

        // A is playing.
        let mut a = SpectrumSource::new(
            Pattern { data: interleaved(FRAMES), at: 0, channels: 2 },
            Arc::clone(&mono), Arc::clone(&stereo),
        );
        for _ in 0..(BATCH_SIZE * 4) {
            a.next();
        }
        let a_gen = stereo.lock().unwrap().generation;
        assert!(!stereo.lock().unwrap().frames.is_empty());

        // B is appended, early. Constructing it must change nothing at all.
        let mut b = SpectrumSource::new(
            Pattern { data: interleaved(FRAMES), at: 0, channels: 2 },
            Arc::clone(&mono), Arc::clone(&stereo),
        );
        assert_eq!(
            stereo.lock().unwrap().generation, a_gen,
            "appending B took the display from A before B was audible"
        );

        // A keeps playing, and keeps the display.
        for _ in 0..(BATCH_SIZE * 4) {
            a.next();
        }
        assert_eq!(stereo.lock().unwrap().generation, a_gen);

        // Rollover: the mixer pulls B for the first time.
        b.next();
        let b_gen = stereo.lock().unwrap().generation;
        assert_ne!(b_gen, a_gen, "B never took the display");
        assert_eq!(stereo.lock().unwrap().channels, 2);

        // Exactly once: further pulls must not keep re-claiming.
        for _ in 0..(BATCH_SIZE * 4) {
            b.next();
        }
        assert_eq!(
            stereo.lock().unwrap().generation, b_gen,
            "B re-claimed the buffer on a later pull"
        );

        // And A being dropped at the rollover must not blank B.
        drop(a);
        let g = stereo.lock().unwrap();
        assert_eq!(g.generation, b_gen, "dropping A revoked its successor");
        assert_eq!(g.channels, 2, "dropping A retracted B's claim");
    }

    /// Dropping the owning source is what gives the lease back.
    #[test]
    fn dropping_the_owner_retires_the_lease() {
        const FRAMES: usize = 8192;
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let mut a = SpectrumSource::new(
            Pattern { data: interleaved(FRAMES), at: 0, channels: 2 },
            Arc::clone(&mono), Arc::clone(&stereo),
        );
        for _ in 0..(BATCH_SIZE * 4) {
            a.next();
        }
        assert_eq!(stereo.lock().unwrap().channels, 2);
        drop(a);
        assert_eq!(
            stereo.lock().unwrap().channels, channels::NO_LIVE_TAP,
            "the lease outlived the source that held it"
        );
    }

    /// A queued source the user skipped past never claimed, so dropping it must
    /// not disturb whatever is actually playing.
    #[test]
    fn dropping_an_unpulled_source_disturbs_nothing() {
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let live = begin_stereo_stream(&stereo, 2);
        let queued = SpectrumSource::new(
            Pattern { data: interleaved(1024), at: 0, channels: 2 },
            Arc::clone(&mono), Arc::clone(&stereo),
        );
        drop(queued);
        let g = stereo.lock().unwrap();
        assert_eq!(g.generation, live);
        assert_eq!(g.channels, 2);
    }

    /// The whole tap path, from an untouched source through the first flush and
    /// well past it, must not allocate — the claim included.
    ///
    /// The shared-buffer capacity check alone missed this: `stereo_batch` was
    /// built with room for `BATCH_SIZE / 2` but fills to `BATCH_SIZE` before
    /// anything flushes it, so it grew on its own first flush, on the thread
    /// feeding the device.
    #[test]
    fn the_source_tap_allocates_on_no_flush_including_the_first() {
        // Enough for many flushes; two channels, so both batches are exercised.
        const FRAMES: usize = BATCH_SIZE * 16;
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let src = Pattern { data: interleaved(FRAMES), at: 0, channels: 2 };
        // Construction may allocate — that is its job, and it happens on the
        // thread assembling the chain.
        let mut tap = SpectrumSource::new(src, Arc::clone(&mono), Arc::clone(&stereo));

        // Armed before the very first pull, so the claim and the first flush
        // are both inside the measurement.
        let before = crate::alloc_probe::arm();
        let mut pulled = 0usize;
        while tap.next().is_some() {
            pulled += 1;
        }
        let allocs = crate::alloc_probe::disarm(before);

        assert_eq!(allocs, 0, "the tap allocated {allocs} time(s) while streaming");
        assert_eq!(pulled, FRAMES * 2, "the measurement must have moved audio");
        assert!(!stereo.lock().unwrap().frames.is_empty(), "and fed the analyser");
    }

    /// Steady state on its own, so a regression that only appears after the
    /// first flush is still caught.
    #[test]
    fn the_source_tap_allocates_on_no_later_flush_either() {
        const FRAMES: usize = BATCH_SIZE * 16;
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let src = Pattern { data: interleaved(FRAMES), at: 0, channels: 2 };
        let mut tap = SpectrumSource::new(src, Arc::clone(&mono), Arc::clone(&stereo));

        // Past the claim and several flushes.
        for _ in 0..(BATCH_SIZE * 6) {
            tap.next();
        }
        let before = crate::alloc_probe::arm();
        for _ in 0..(BATCH_SIZE * 20) {
            tap.next();
        }
        let allocs = crate::alloc_probe::disarm(before);
        assert_eq!(allocs, 0, "steady-state streaming allocated {allocs} time(s)");
    }

    /// A mono source claims one channel at its first pull, not at construction.
    #[test]
    fn a_mono_source_claims_one_channel_when_pulled() {
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let src = Pattern { data: vec![0.25; 2048], at: 0, channels: 1 };
        let mut tapped = SpectrumSource::new(src, Arc::clone(&mono), Arc::clone(&stereo));
        assert_eq!(
            stereo.lock().unwrap().channels, channels::NO_LIVE_TAP,
            "construction must not claim"
        );
        tapped.next();
        assert_eq!(stereo.lock().unwrap().channels, 1);
        while tapped.next().is_some() {}
        assert!(
            stereo.lock().unwrap().frames.is_empty(),
            "a mono source must not fabricate stereo frames"
        );
    }

    /// The stereo history has to cover the largest window the FFT-size selector
    /// offers, and the flush must not be the thing that grows it.
    #[test]
    fn the_stereo_history_covers_the_largest_analysis_window() {
        // That the caps clear the window is asserted at compile time beside
        // the constants. What needs a running tap is the rest: that the
        // constant really is the largest window anything asks for, and that
        // filling past it does not grow the buffer.
        assert_eq!(MAX_ANALYSIS_WINDOW, auto_fft_size_for(768_000));
        assert_eq!(MAX_ANALYSIS_WINDOW, auto_fft_size_for(192_000));

        let stereo = new_stereo_buf();
        begin_stereo_stream(&stereo, 2);
        let cap_after_claim = stereo.lock().unwrap().frames.capacity();
        assert!(cap_after_claim >= STEREO_CAP, "the claim must reserve the cap");

        // Fill well past the cap through the ordinary tap, then confirm the
        // capacity never moved — the flush trims before it extends.
        let mono = new_sample_buf();
        let frames = STEREO_CAP + 5000;
        let src = Pattern { data: interleaved(frames), at: 0, channels: 2 };
        let mut tapped = SpectrumSource::new(src, Arc::clone(&mono), Arc::clone(&stereo));
        let cap_before = stereo.lock().unwrap().frames.capacity();
        while tapped.next().is_some() {}
        let g = stereo.lock().unwrap();
        assert_eq!(
            g.frames.capacity(), cap_before,
            "the flush reallocated the stereo buffer"
        );
        assert!(g.frames.len() <= STEREO_CAP, "the cap was exceeded");
        assert!(g.frames.len() >= MAX_ANALYSIS_WINDOW, "not enough history kept");
    }
}

// ---------------------------------------------------------------------------
// The window's own channel adapter
// ---------------------------------------------------------------------------

/// `tick_channels` is the route a user actually takes: it reads the shared tap,
/// decides availability from the channel count published there, deinterleaves,
/// and hands the result to the analyser. Testing `process_channels` alone left
/// that whole adapter uncovered — a mutant that averaged the pair into both
/// buffers passed the entire suite, because every channel test fed the analyser
/// two ready-made buffers and never went through the step that builds them.
#[cfg(test)]
mod channel_adapter_tests {
    use super::*;

    const FFT: usize = 1024;

    fn window(view: channels::ChannelView, style: VizStyle) -> SpectrumWindow {
        let mut w = SpectrumWindow::new();
        w.mode = SpectrumMode::RealTime;
        w.style = style;
        w.channel_view = view;
        w.analyzer.sample_rate = 48_000;
        w.analyzer.fft_size = FFT;
        w.analyzer.smoothing = 0.0;
        w.analyzer.min_freq = 20.0;
        w.analyzer.max_freq = 20_000.0;
        w.analyzer.bar_count = 256;
        w.analyzer.rebuild_fft();
        w.analyzer.magnitudes = vec![0.0; 256];
        w.analyzer.smoothed = vec![0.0; 256];
        w.analyzer.peak_input = vec![0.0; 256];
        w
    }

    /// Fill the shared tap the way a real stream would, through the same
    /// claim/flush seam the taps use.
    fn feed(w: &SpectrumWindow, channels_n: u16, left: &[f32], right: &[f32]) {
        begin_stereo_stream(&w.stereo_buf, channels_n);
        let mut g = w.stereo_buf.lock().unwrap();
        for (l, r) in left.iter().zip(right.iter()) {
            g.frames.push([*l, *r]);
        }
    }

    fn tone(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / 48_000.0).sin() * 0.5)
            .collect()
    }

    fn peak_bar(bars: &[f32]) -> usize {
        bars.iter()
            .enumerate()
            .fold((0usize, 0.0f32), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) })
            .0
    }

    /// The whole route: tap in, two different spectra out, in the right order.
    #[test]
    fn the_adapter_produces_the_two_channels_it_was_given() {
        let mut w = window(channels::ChannelView::Split, VizStyle::Bars);
        feed(&w, 2, &tone(500.0, FFT * 2), &tone(5000.0, FFT * 2));
        w.tick_channels(1.0 / 60.0);

        assert_eq!(
            w.channel_availability,
            channels::ChannelAvailability::Available
        );
        let frame = w.analyzer.channel_frame();
        let (l, r) = frame.pair().expect("a two-channel tap must yield a pair");
        let (lp, rp) = (peak_bar(l), peak_bar(r));
        assert!(lp < rp, "left peaked at {lp}, right at {rp} — 500 Hz is below 5 kHz");
        // The mutation this test exists for: averaging the pair into both
        // buffers gives two identical spectra, each with two peaks.
        assert!(l != r, "the two channels must not be the same series");
        assert!(
            r[lp] < l[lp] * 0.25,
            "the right channel carries the left's 500 Hz energy — the pair was blended"
        );
        assert!(
            l[rp] < r[rp] * 0.25,
            "the left channel carries the right's 5 kHz energy — the pair was blended"
        );
    }

    /// Mix must take none of that path, and leave nothing for the renderer.
    #[test]
    fn mix_leaves_the_adapter_idle() {
        let mut w = window(channels::ChannelView::Mix, VizStyle::Bars);
        feed(&w, 2, &tone(500.0, FFT * 2), &tone(5000.0, FFT * 2));
        let _ = w.analyzer.take_channel_ffts();
        w.tick_channels(1.0 / 60.0);
        assert_eq!(w.analyzer.take_channel_ffts(), 0);
        assert!(w.analyzer.channel_frame().pair().is_none());
    }

    /// A mono stream must not produce channels however hard the view asks.
    #[test]
    fn a_mono_stream_gives_the_adapter_nothing_to_split() {
        let mut w = window(channels::ChannelView::Overlay, VizStyle::Line);
        // A mono tap pushes no pairs at all; the buffer is claimed at 1 channel.
        begin_stereo_stream(&w.stereo_buf, 1);
        w.tick_channels(1.0 / 60.0);
        assert_eq!(w.channel_availability, channels::ChannelAvailability::Mono);
        assert!(w.analyzer.channel_frame().pair().is_none());
        assert_eq!(
            channels::effective_view(w.channel_view, &w.channel_availability),
            channels::ChannelView::Mix
        );
    }

    /// Surround material must not be paired off as L/R.
    #[test]
    fn multichannel_material_is_refused_by_the_adapter() {
        let mut w = window(channels::ChannelView::Split, VizStyle::Bars);
        feed(&w, 6, &tone(500.0, FFT * 2), &tone(5000.0, FFT * 2));
        w.tick_channels(1.0 / 60.0);
        assert_eq!(w.channel_availability, channels::ChannelAvailability::Multichannel(6));
        assert!(w.analyzer.channel_frame().pair().is_none());
    }

    /// The native-DSD shape: a populated buffer from the previous track, and
    /// nothing feeding it now. The frames are still there; the claim is not.
    #[test]
    fn a_stale_buffer_cannot_enable_left_and_right() {
        let mut w = window(channels::ChannelView::Split, VizStyle::Bars);
        feed(&w, 2, &tone(500.0, FFT * 2), &tone(5000.0, FFT * 2));
        w.tick_channels(1.0 / 60.0);
        assert!(w.analyzer.channel_frame().pair().is_some());

        // The next track is native DSD: nothing attaches a tap.
        invalidate_live_pcm_route(&w.stereo_buf, &w.sample_buf);
        w.tick_channels(1.0 / 60.0);
        assert_eq!(
            w.channel_availability,
            channels::ChannelAvailability::NoLiveTap
        );
        assert!(
            w.analyzer.channel_frame().pair().is_none(),
            "the previous track's frames were shown as this track's channels"
        );
    }

    /// A visualisation with no per-channel form says which obstacle it is,
    /// rather than going blank or quietly drawing the mix twice.
    #[test]
    fn an_unsupported_style_reports_itself_through_the_adapter() {
        let mut w = window(channels::ChannelView::Split, VizStyle::Waterfall);
        feed(&w, 2, &tone(500.0, FFT * 2), &tone(5000.0, FFT * 2));
        w.tick_channels(1.0 / 60.0);
        assert_eq!(w.channel_availability, channels::ChannelAvailability::UnsupportedStyle);
        assert!(w.analyzer.channel_frame().pair().is_none());
        assert!(w.channel_availability.reason().is_some());
        // And the preference survives, so switching back to Bars restores it.
        assert_eq!(w.channel_view, channels::ChannelView::Split);
        w.style = VizStyle::Bars;
        w.tick_channels(1.0 / 60.0);
        assert_eq!(
            w.channel_availability,
            channels::ChannelAvailability::Available
        );
        assert!(w.analyzer.channel_frame().pair().is_some());
    }

    /// Losing availability must clear the spectra, not freeze the last frame.
    #[test]
    fn losing_availability_resets_the_channel_smoothers() {
        let mut w = window(channels::ChannelView::Split, VizStyle::Bars);
        w.analyzer.smoothing = 0.9;
        feed(&w, 2, &tone(500.0, FFT * 2), &tone(5000.0, FFT * 2));
        for _ in 0..5 {
            w.tick_channels(1.0 / 60.0);
        }
        assert!(w.analyzer.channel_frame().pair().is_some());

        w.channel_view = channels::ChannelView::Mix;
        w.tick_channels(1.0 / 60.0);
        assert!(
            w.analyzer.channel_frame().pair().is_none(),
            "switching to Mix must not leave the old channel bars behind"
        );
    }

    /// Pre-process names the cache as the reason, not the tap — the tap may be
    /// perfectly good stereo and still be the wrong place to look.
    #[test]
    fn preprocess_reports_the_cache_as_the_obstacle() {
        let mut w = window(channels::ChannelView::Split, VizStyle::Bars);
        feed(&w, 2, &tone(500.0, FFT * 2), &tone(5000.0, FFT * 2));
        w.mode = SpectrumMode::PreProcess;
        w.channel_availability = channels::availability(
            true, w.style_supports_channels(), channels::NO_LIVE_TAP,
        );
        assert_eq!(w.channel_availability, channels::ChannelAvailability::PreProcessMono);
        assert!(
            w.channel_availability
                .reason()
                .is_some_and(|r| r.contains("channel-aware cache"))
        );
    }
}

// ---------------------------------------------------------------------------
// Mode switching, and telling the truth while nothing is playing
// ---------------------------------------------------------------------------

/// Real-time and Pre-process read unrelated producers, so moving between them
/// is a discontinuity — and the display has to keep saying what it can show
/// even when the transport has stopped.
#[cfg(test)]
mod mode_and_availability_tests {
    use super::*;

    const RATE: f64 = 180.0;
    const BARS: usize = 8;
    const FFT: usize = 1024;

    fn cache(n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|f| {
                (0..BARS)
                    .map(|b| (((f * 7 + b * 13) % 97) as f32) / 97.0)
                    .collect()
            })
            .collect()
    }

    fn window() -> SpectrumWindow {
        let mut w = SpectrumWindow::new();
        w.analyzer.sample_rate = 48_000;
        w.analyzer.fft_size = FFT;
        w.analyzer.bar_count = BARS;
        w.bar_count = BARS;
        w.analyzer.smoothing = 0.75;
        w.analyzer.rebuild_fft();
        w.analyzer.magnitudes = vec![0.0; BARS];
        w.analyzer.smoothed = vec![0.0; BARS];
        w.analyzer.peak_input = vec![0.0; BARS];
        w.analyzer.waterfall_enabled = true;
        w.style = VizStyle::Bars;
        w
    }

    fn tone(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / 48_000.0).sin() * 0.5)
            .collect()
    }

    /// A tick the `max_fps` throttle will accept.
    ///
    /// These run microseconds apart, where a real UI is at least milliseconds
    /// apart, so the throttle would swallow every tick after the first and the
    /// test would prove nothing about the code underneath it.
    fn tick_now(w: &mut SpectrumWindow, at: f64, playing: bool) {
        w.last_fft_time = None;
        w.tick(at, playing);
    }

    fn feed_stereo(w: &SpectrumWindow, channels_n: u16, n: usize) {
        begin_stereo_stream(&w.stereo_buf, channels_n);
        let (l, r) = (tone(500.0, n), tone(5000.0, n));
        let mut g = w.stereo_buf.lock().unwrap();
        for i in 0..n {
            g.frames.push([l[i], r[i]]);
        }
    }

    /// Switching away and back must not replay the rows the clock crossed while
    /// the other mode was showing. Without a reset the cursor still pointed at
    /// where Pre-process left off, and returning to it cascaded every row in
    /// between — a burst of history nobody played.
    #[test]
    fn switching_modes_does_not_replay_cached_history() {
        let mut w = window();
        w.mode = SpectrumMode::PreProcess;
        // `tick` derives `waterfall_enabled` from the style, so the rolling
        // history is only actually produced when it is the visualisation. With
        // Bars selected the waterfall assertions below would pass against any
        // implementation at all.
        w.style = VizStyle::Waterfall;
        w.analyzer.set_pre_frames(cache(4096), RATE);

        // Play a little in Pre-process.
        for k in 0..=10 {
            tick_now(&mut w, k as f64 / 60.0, true);
        }
        let cursor = w.analyzer.last_pre_frame;
        assert!(cursor.is_some(), "setup: the cursor should have advanced");
        assert!(
            !w.analyzer.waterfall.is_empty(),
            "setup: the waterfall should have rows to lose"
        );

        // Switch to Real-time. The cursor must not survive.
        w.mode = SpectrumMode::RealTime;
        tick_now(&mut w, 10.0 / 60.0, true);
        assert_eq!(
            w.analyzer.last_pre_frame, None,
            "the pre-process cursor survived a switch to Real-time"
        );
        assert!(
            w.analyzer.waterfall.is_empty(),
            "waterfall rows from the other mode survived the switch"
        );

        // Five seconds of Real-time pass, then back to Pre-process.
        let rows_before = w.analyzer.waterfall_seq;
        w.mode = SpectrumMode::PreProcess;
        tick_now(&mut w, 5.0, true);

        // One row consumed — the snap — not the 900 the clock crossed.
        assert_eq!(
            w.analyzer.last_pre_frame,
            Some(timing::target_frame(5.0, RATE, 4096).unwrap()),
            "returning to Pre-process must snap to the clock"
        );
        assert!(
            w.analyzer.waterfall_seq - rows_before <= 2,
            "returning to Pre-process replayed {} waterfall rows",
            w.analyzer.waterfall_seq - rows_before
        );
    }

    /// And the same in the other direction: Real-time state must not leak into
    /// a Pre-process cascade.
    #[test]
    fn switching_into_preprocess_snaps_rather_than_cascading() {
        let mut w = window();
        w.mode = SpectrumMode::RealTime;
        w.analyzer.set_pre_frames(cache(4096), RATE);
        for k in 0..=10 {
            tick_now(&mut w, k as f64 / 60.0, true);
        }
        assert_eq!(w.analyzer.last_pre_frame, None);

        w.mode = SpectrumMode::PreProcess;
        let frames = cache(4096);
        w.analyzer.smoothing = 0.0;
        tick_now(&mut w, 300.0 / RATE, true);
        assert_eq!(
            w.analyzer.magnitudes, frames[300],
            "the first Pre-process tick must show the row the clock points at"
        );
    }

    /// A stopped player must not go on offering Left/Right, and must not leave
    /// the last live frame on screen. Stopping retires the lease; the display
    /// has to notice even though nothing is playing.
    #[test]
    fn a_stopped_player_reports_no_live_tap() {
        let mut w = window();
        w.mode = SpectrumMode::RealTime;
        w.channel_view = channels::ChannelView::Split;
        feed_stereo(&w, 2, FFT * 2);
        tick_now(&mut w, 0.0, true);
        assert_eq!(
            w.channel_availability,
            channels::ChannelAvailability::Available
        );
        assert!(w.analyzer.channel_frame().pair().is_some());

        // The engine tears the route down and retires the lease.
        invalidate_live_pcm_route(&w.stereo_buf, &w.sample_buf);
        // A tick with nothing playing — which used to return before it looked.
        tick_now(&mut w, 1.0, false);

        assert_eq!(
            w.channel_availability,
            channels::ChannelAvailability::NoLiveTap
        );
        assert!(
            w.analyzer.channel_frame().pair().is_none(),
            "a stopped player kept a live-looking left and right"
        );
        assert!(w.channel_availability.reason().is_some());
    }

    /// Changing visualisation while paused takes effect while paused.
    #[test]
    fn a_style_change_while_paused_updates_availability() {
        let mut w = window();
        w.mode = SpectrumMode::RealTime;
        w.channel_view = channels::ChannelView::Overlay;
        feed_stereo(&w, 2, FFT * 2);
        tick_now(&mut w, 0.0, true);
        assert_eq!(
            w.channel_availability,
            channels::ChannelAvailability::Available
        );

        w.style = VizStyle::Waterfall;
        tick_now(&mut w, 0.0, false);
        assert_eq!(
            w.channel_availability,
            channels::ChannelAvailability::UnsupportedStyle,
            "a style change while paused was not noticed"
        );
        assert!(w.analyzer.channel_frame().pair().is_none());

        w.style = VizStyle::Line;
        tick_now(&mut w, 0.0, false);
        assert_eq!(
            w.channel_availability,
            channels::ChannelAvailability::Available
        );
    }

    /// And changing mode while paused, which is the case that used to leave the
    /// controls enabled against a cache that cannot supply channels.
    #[test]
    fn a_mode_change_while_paused_updates_availability() {
        let mut w = window();
        w.mode = SpectrumMode::RealTime;
        w.channel_view = channels::ChannelView::Split;
        feed_stereo(&w, 2, FFT * 2);
        tick_now(&mut w, 0.0, true);
        assert_eq!(
            w.channel_availability,
            channels::ChannelAvailability::Available
        );

        w.mode = SpectrumMode::PreProcess;
        tick_now(&mut w, 0.0, false);
        assert_eq!(
            w.channel_availability,
            channels::ChannelAvailability::PreProcessMono,
            "switching to Pre-process while paused left Left/Right enabled"
        );
        assert!(w.analyzer.channel_frame().pair().is_none());
        assert!(
            w.channel_availability
                .reason()
                .is_some_and(|r| r.contains("channel-aware cache"))
        );
    }

    /// Mono and multichannel are noticed while paused too — a track change can
    /// happen without the transport running.
    #[test]
    fn mono_and_multichannel_are_noticed_while_paused() {
        for (n, want) in [
            (1u16, channels::ChannelAvailability::Mono),
            (6u16, channels::ChannelAvailability::Multichannel(6)),
        ] {
            let mut w = window();
            w.mode = SpectrumMode::RealTime;
            w.channel_view = channels::ChannelView::Left;
            feed_stereo(&w, 2, FFT * 2);
            tick_now(&mut w, 0.0, true);
            assert_eq!(
            w.channel_availability,
            channels::ChannelAvailability::Available
        );

            begin_stereo_stream(&w.stereo_buf, n);
            tick_now(&mut w, 0.0, false);
            assert_eq!(w.channel_availability, want);
            assert!(w.analyzer.channel_frame().pair().is_none());
        }
    }

    /// One row per analysis update, at every rate, in both modes.
    ///
    /// The name and doc this test carried — "the lesser of target and producer"
    /// — described the `WATERFALL_TARGET_HZ` cap, which was deleted when the
    /// owner reported the waterfall scrolling three times slower than 1.4.5.
    /// There is no target any more and no lesser-of anything; the body was
    /// already asserting the current rule while the name went on describing the
    /// removed one.
    #[test]
    fn every_analysis_update_produces_exactly_one_row() {
        // (producer updates per second, expected rows in one second)
        //
        // One row per update, at every rate. A previous release capped this at
        // 60, which on a ~180 Hz display made the waterfall scroll three times
        // slower than the release before it and, in Pre-process, threw away two
        // of every three analysed rows — the bars showed them, the history did
        // not.
        let rates = [1.0f64, 24.0, 30.0, 46.875, 60.0, 144.0, 180.0, 240.0];
        for producer_hz in rates {
            let mut w = window();
            w.mode = SpectrumMode::RealTime;
            w.analyzer.waterfall_enabled = true;
            // Deep enough that the ring is not the thing under test.
            w.analyzer.waterfall_rows = 4096;
            {
                let mut b = w.sample_buf.lock().unwrap();
                b.extend_from_slice(&tone(1000.0, FFT * 2));
            }
            let seq0 = w.analyzer.waterfall_seq;
            let ticks = producer_hz.round() as usize;
            for _ in 0..ticks {
                w.analyzer.process_realtime(1.0 / producer_hz);
            }
            let rows = (w.analyzer.waterfall_seq - seq0) as f64;
            assert!(
                (rows - ticks as f64).abs() < 0.5,
                "a {producer_hz} Hz producer gave {rows} rows for {ticks} updates"
            );
        }
    }

    /// Nothing the cursor crosses is dropped from the history.
    ///
    /// The bars and the waterfall are fed from the same rows, so a transient
    /// that reaches one has to reach the other. Under the cap, a 180 fps cache
    /// against any display put every third row in the waterfall and all of them
    /// in the bars.
    #[test]
    fn the_waterfall_receives_every_consumed_cache_row() {
        let mut w = window();
        w.mode = SpectrumMode::PreProcess;
        w.style = VizStyle::Waterfall;
        w.analyzer.waterfall_rows = 4096;
        w.analyzer.set_pre_frames(cache(4096), 180.0);

        // One tick that crosses many rows at once, which is what a 60 Hz
        // display against a 180 fps cache does three times a second.
        tick_now(&mut w, 0.0, true);
        let seq0 = w.analyzer.waterfall_seq;
        let cursor0 = w.analyzer.last_pre_frame.expect("a cursor after the first tick");
        tick_now(&mut w, 1.0, true);
        let consumed = w.analyzer.last_pre_frame.unwrap() - cursor0;
        let rows = w.analyzer.waterfall_seq - seq0;
        assert!(consumed > 1, "setup: the tick should cross several rows");
        assert_eq!(
            rows, consumed as u64,
            "{consumed} rows were consumed but {rows} reached the waterfall"
        );
    }

    /// The depth is chosen in seconds and the row count follows the rate, so
    /// the same span costs different amounts of memory in different modes.
    #[test]
    fn the_history_depth_follows_the_producer_rate() {
        // 0.67 s is the shipped default, and what 1.4.5 produced on a ~180 Hz
        // display with its fixed 120-row ring.
        let rows = timing::waterfall_rows_for(timing::DEFAULT_WATERFALL_SECS, 180.0);
        assert_eq!(rows, 121, "0.67 s at 180/s should be ~120 rows, got {rows}");

        // The same span at other rates.
        assert_eq!(timing::waterfall_rows_for(0.67, 60.0), 40);
        assert_eq!(timing::waterfall_rows_for(2.0, 180.0), 360);

        // Bounded at both ends, because every row costs bars on the CPU and a
        // texture row on the GPU.
        assert_eq!(timing::waterfall_rows_for(0.001, 1.0), 32);
        assert_eq!(timing::waterfall_rows_for(8.0, 100_000.0), 2048);
        // And degenerate input does not produce a zero-height ring.
        assert_eq!(timing::waterfall_rows_for(f32::NAN, 180.0), 32);
        assert_eq!(timing::waterfall_rows_for(1.0, 0.0), 32);
    }

    /// Where a row bound binds, the span on screen is not the span requested —
    /// and the number worth showing is the one the ring actually holds.
    #[test]
    fn a_bound_moves_the_span_off_the_slider() {
        // Neither bound binding: the ring holds what was asked for.
        let rows = timing::waterfall_rows_for(2.0, 180.0);
        let span = timing::waterfall_span_secs(rows, 180.0);
        assert!((span - 2.0).abs() < 0.01, "unbounded span drifted to {span}");

        // The floor binds at a low rate: 0.2 s wants 12 rows and gets 32, which
        // is more than two and a half times the span requested.
        let rows = timing::waterfall_rows_for(0.2, 60.0);
        assert_eq!(rows, timing::WATERFALL_MIN_ROWS);
        let span = timing::waterfall_span_secs(rows, 60.0);
        assert!(
            span > 0.5,
            "the floor should stretch 0.2 s well past it, got {span}"
        );

        // The ceiling binds at a high rate, the other way.
        let rows = timing::waterfall_rows_for(8.0, 400.0);
        assert_eq!(rows, timing::WATERFALL_MAX_ROWS);
        let span = timing::waterfall_span_secs(rows, 400.0);
        assert!(
            span < 6.0,
            "the ceiling should cut 8 s well short of it, got {span}"
        );

        // A rate of nothing has no span rather than an infinite one.
        assert_eq!(timing::waterfall_span_secs(120, 0.0), 0.0);
        assert_eq!(timing::waterfall_span_secs(120, f64::NAN), 0.0);
    }

    /// Pre-process measures its row rate; Real-time only requests one. The
    /// difference is the whole reason the duration is called nominal there, so
    /// the flag that says which has to follow the mode.
    #[test]
    fn only_preprocess_knows_its_row_rate() {
        let mut w = window();

        // Real-time: Max FPS is a ceiling asked of the analyser, and nothing
        // here measures what it achieved.
        w.mode = SpectrumMode::RealTime;
        w.max_fps = 144.0;
        let (hz, nominal) = w.waterfall_row_rate();
        assert_eq!(hz, 144.0);
        assert!(nominal, "a Max FPS ceiling is not a measured rate");

        // Pre-process with a cache: every cached row is consumed, so the
        // cache's own frame rate is the rate, measured.
        w.mode = SpectrumMode::PreProcess;
        w.analyzer.set_pre_frames(cache(512), RATE);
        let (hz, nominal) = w.waterfall_row_rate();
        assert_eq!(hz, RATE);
        assert!(!nominal, "the cache frame rate is a measurement");

        // Pre-process with no cache yet has nothing to measure, so it falls
        // back to the request and must say so. A fresh window rather than
        // clearing this one: `pre_frames` has a single sanctioned clearing
        // path and `nothing_writes_the_cached_matrix_directly` holds it to
        // that, correctly.
        let mut empty = window();
        empty.mode = SpectrumMode::PreProcess;
        assert!(empty.analyzer.pre_frames.is_empty(), "setup: no cache");
        let (_, nominal) = empty.waterfall_row_rate();
        assert!(nominal, "with no cache there is no measured rate to report");
    }

    /// The window keeps the ring in step with whatever is feeding it.
    #[test]
    fn changing_the_span_or_the_rate_resizes_the_ring() {
        let mut w = window();
        w.mode = SpectrumMode::RealTime;
        w.max_fps = 180.0;
        w.waterfall_secs = 0.67;
        w.tick(0.0, false);
        let at_180 = w.analyzer.waterfall_rows;
        assert_eq!(at_180, 121);

        // A slower display: the same span, fewer rows.
        w.max_fps = 60.0;
        w.tick(0.0, false);
        assert_eq!(w.analyzer.waterfall_rows, 40);

        // A longer span: more rows.
        w.waterfall_secs = 4.0;
        w.tick(0.0, false);
        assert_eq!(w.analyzer.waterfall_rows, 240);

        // And the ring is trimmed rather than left above its new cap.
        w.analyzer.waterfall_enabled = true;
        for _ in 0..500 {
            w.analyzer.push_waterfall_for_test(vec![0.5; BARS]);
        }
        assert_eq!(w.analyzer.waterfall.len(), 240);
        // 0.5 s at 60/s is 30 rows, which the floor lifts to 32: below that a
        // rolling history stops being one.
        w.waterfall_secs = 0.5;
        w.tick(0.0, false);
        assert_eq!(w.analyzer.waterfall_rows, 32);
        assert_eq!(
            w.analyzer.waterfall.len(),
            32,
            "the ring was left holding more rows than its new depth"
        );
    }

    /// A stall costs rows, because the waterfall advances per update and a
    /// stall is the absence of updates. Nothing is synthesised to fill it.
    #[test]
    fn a_long_stall_produces_one_row_not_a_burst() {
        let mut w = window();
        w.mode = SpectrumMode::RealTime;
        w.analyzer.waterfall_enabled = true;
        w.analyzer.waterfall_rows = 4096;
        {
            let mut b = w.sample_buf.lock().unwrap();
            b.extend_from_slice(&tone(1000.0, FFT * 2));
        }
        let seq0 = w.analyzer.waterfall_seq;
        // Five seconds pass with a single update at the end of it.
        w.analyzer.process_realtime(5.0);
        assert_eq!(
            w.analyzer.waterfall_seq - seq0,
            1,
            "a stall is one update, so it is one row"
        );

        // And the rate afterwards is the producer's, not a catch-up burst.
        let seq1 = w.analyzer.waterfall_seq;
        for _ in 0..240 {
            w.analyzer.process_realtime(1.0 / 240.0);
        }
        assert_eq!(w.analyzer.waterfall_seq - seq1, 240);
    }

    /// Two producers of the same rate give the same axis whichever mode they
    /// are in, which is the part of the claim that does hold unqualified.
    #[test]
    fn both_modes_agree_at_the_same_producer_rate() {
        let mut rt = window();
        rt.mode = SpectrumMode::RealTime;
        rt.analyzer.waterfall_enabled = true;
        {
            let mut b = rt.sample_buf.lock().unwrap();
            b.extend_from_slice(&tone(1000.0, FFT * 2));
        }
        let rt0 = rt.analyzer.waterfall_seq;
        for _ in 0..180 {
            rt.analyzer.process_realtime(1.0 / 180.0);
        }
        let rt_rows = rt.analyzer.waterfall_seq - rt0;

        let mut pre = window();
        pre.mode = SpectrumMode::PreProcess;
        pre.style = VizStyle::Waterfall;
        pre.analyzer.set_pre_frames(cache(4096), 180.0);
        let pre0 = pre.analyzer.waterfall_seq;
        for k in 0..=180 {
            tick_now(&mut pre, k as f64 / 180.0, true);
        }
        let pre_rows = pre.analyzer.waterfall_seq - pre0;

        assert!(
            rt_rows.abs_diff(pre_rows) <= 2,
            "180 Hz producers disagreed: Real-time {rt_rows}, Pre-process {pre_rows}"
        );
    }
}

// ---------------------------------------------------------------------------
// Forced schedules: a late first pull against a route that has ended
// ---------------------------------------------------------------------------

/// `Sink::stop` sets an atomic the mixer notices asynchronously, so a render
/// callback that has already passed the stop check can still enter a source
/// belonging to a session the engine has finished with. The claim is
/// unconditional by construction — it clears the buffer and takes the
/// generation — so nothing in the write path could refuse it. These drive the
/// orderings that actually arise, rather than asserting the protocol from the
/// outside.
#[cfg(test)]
mod route_epoch_tests {
    use super::stereo_tap_tests::{Pattern, interleaved};
    use super::*;

    const FRAMES: usize = 8192;

    fn source(stereo: &StereoBuf, mono: &SampleBuf) -> SpectrumSource<Pattern> {
        SpectrumSource::new(
            Pattern {
                data: interleaved(FRAMES),
                at: 0,
                channels: 2,
            },
            Arc::clone(mono),
            Arc::clone(stereo),
        )
    }

    /// Distinctive frames, so "the new route's data is still there" is a real
    /// assertion rather than "the buffer is non-empty".
    fn sentinels(stereo: &StereoBuf, n: usize) -> Vec<[f32; 2]> {
        let rows: Vec<[f32; 2]> = (0..n)
            .map(|i| [1000.0 + i as f32, -(1000.0 + i as f32)])
            .collect();
        let mut g = stereo.lock().unwrap();
        g.frames.extend_from_slice(&rows);
        rows
    }

    /// Schedule 1: the source is constructed, never pulled, the route ends, a
    /// new route claims and writes — and only then is the old source pulled.
    ///
    /// Without an epoch its first pull claims unconditionally: it would clear
    /// the successor's frames and take the generation.
    #[test]
    fn a_never_pulled_source_cannot_claim_after_its_route_ended() {
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let mut old = source(&stereo, &mono);

        // The engine tears the route down.
        invalidate_live_pcm_route(&stereo, &mono);
        // A new route starts and writes.
        let new_gen = begin_stereo_stream(&stereo, 2);
        let marks = sentinels(&stereo, 64);
        let new_epoch = stereo.lock().unwrap().route_epoch;

        // Now the mixer finally enters the old source.
        for _ in 0..(BATCH_SIZE * 8) {
            old.next();
        }

        let g = stereo.lock().unwrap();
        assert_eq!(
            g.generation, new_gen,
            "the stale source took the generation"
        );
        assert_eq!(g.route_epoch, new_epoch, "the stale source moved the epoch");
        assert_eq!(g.channels, 2, "the stale source retracted the live claim");
        assert_eq!(g.frames, marks, "the stale source erased the live frames");
    }

    /// And it must not merely fail once: a `Stale` result is final. A source
    /// that retried would eventually succeed against whatever route is running.
    #[test]
    fn a_stale_source_never_retries_into_a_later_route() {
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let mut old = source(&stereo, &mono);
        invalidate_live_pcm_route(&stereo, &mono);

        // First pull: refused, permanently.
        old.next();

        // Several further routes come and go while the old source drains.
        for _ in 0..3 {
            let owner = begin_stereo_stream(&stereo, 2);
            let marks = sentinels(&stereo, 32);
            for _ in 0..(BATCH_SIZE * 4) {
                old.next();
            }
            let g = stereo.lock().unwrap();
            assert_eq!(g.generation, owner, "a stale source claimed a later route");
            assert_eq!(g.frames, marks, "a stale source wrote into a later route");
            drop(g);
            invalidate_live_pcm_route(&stereo, &mono);
        }
        // And dropping it takes nothing with it.
        let before = stereo.lock().unwrap().generation;
        drop(old);
        assert_eq!(stereo.lock().unwrap().generation, before);
    }

    /// `Stale` has to be *final*, not merely unsuccessful.
    ///
    /// With the epoch check in place a retry cannot succeed while the epoch
    /// keeps moving forward, so "retries for ever" and "stops trying" look the
    /// same from outside — one of them just burns a `try_lock` per sample on
    /// the audio thread. This forces the difference into the open by putting
    /// the epoch back to the value the source captured, which is the strongest
    /// form of "a later route it must not claim": a source that merely stopped
    /// succeeding would take it, and one that stopped trying will not.
    #[test]
    fn a_stale_source_stays_stale_even_if_its_epoch_comes_back() {
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let captured = stereo.lock().unwrap().route_epoch;
        let mut old = source(&stereo, &mono);

        invalidate_live_pcm_route(&stereo, &mono);
        old.next(); // refused — and the refusal must be permanent

        // A later route that happens to present the same epoch.
        {
            let mut g = stereo.lock().unwrap();
            g.route_epoch = captured;
        }
        let live = begin_stereo_stream(&stereo, 2);
        let marks = sentinels(&stereo, 32);
        {
            let mut g = stereo.lock().unwrap();
            g.route_epoch = captured;
        }

        for _ in 0..(BATCH_SIZE * 8) {
            old.next();
        }

        let g = stereo.lock().unwrap();
        assert_eq!(
            g.generation, live,
            "a stale source claimed a route that presented its old epoch"
        );
        assert_eq!(
            g.frames, marks,
            "a stale source wrote after being refused once"
        );
    }

    /// Schedule 2: the old source's first pull is parked against the teardown —
    /// it tries while the lock is held, gets `Busy`, and by the time it can
    /// retry the successor owns the buffer.
    #[test]
    fn a_first_pull_parked_across_teardown_is_rejected_not_retried() {
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let mut old = source(&stereo, &mono);

        // Park it: the lock is held, so the claim can only return Busy.
        {
            let _held = stereo.lock().unwrap();
            old.next();
        }
        // The route ends and a successor takes over while it was parked.
        invalidate_live_pcm_route(&stereo, &mono);
        let new_gen = begin_stereo_stream(&stereo, 2);
        let marks = sentinels(&stereo, 48);

        for _ in 0..(BATCH_SIZE * 8) {
            old.next();
        }
        let g = stereo.lock().unwrap();
        assert_eq!(g.generation, new_gen);
        assert_eq!(
            g.frames, marks,
            "the parked source wrote after its route ended"
        );
    }

    /// A `Busy` claim within the *same* route must still succeed later — the
    /// retry path has to stay alive for the case it exists for.
    #[test]
    fn a_busy_claim_retries_within_the_same_route() {
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let mut src = source(&stereo, &mono);
        {
            let _held = stereo.lock().unwrap();
            src.next();
        }
        // Nothing ended; the next pull should get it.
        src.next();
        let g = stereo.lock().unwrap();
        assert_eq!(g.channels, 2, "a busy claim never retried");
    }

    /// Schedule 3: A playing and B queued belong to one session, so B can claim
    /// at its first real pull even though it was constructed seconds earlier.
    #[test]
    fn a_queued_source_shares_the_route_epoch_with_the_one_playing() {
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();

        let mut a = source(&stereo, &mono);
        for _ in 0..(BATCH_SIZE * 4) {
            a.next();
        }
        let a_gen = stereo.lock().unwrap().generation;
        let epoch = stereo.lock().unwrap().route_epoch;

        // B is appended to the same sink, mid-track.
        let mut b = source(&stereo, &mono);
        {
            let g = stereo.lock().unwrap();
            assert_eq!(g.generation, a_gen, "constructing B disturbed A");
            assert_eq!(g.route_epoch, epoch, "B was given a different session");
        }

        // Rollover.
        b.next();
        let b_gen = stereo.lock().unwrap().generation;
        assert_ne!(b_gen, a_gen, "B could not claim within its own session");

        // Exactly once.
        for _ in 0..(BATCH_SIZE * 4) {
            b.next();
        }
        assert_eq!(stereo.lock().unwrap().generation, b_gen, "B re-claimed");

        // A is dropped at the rollover and must not blank B, nor end the
        // session B is still playing in.
        drop(a);
        let g = stereo.lock().unwrap();
        assert_eq!(g.generation, b_gen, "dropping A revoked its successor");
        assert_eq!(g.channels, 2);
        assert_eq!(g.route_epoch, epoch, "dropping A ended the shared session");
    }

    /// A source `Drop` is one owner leaving, never the session ending — so a
    /// source constructed afterwards, in the same session, can still claim.
    #[test]
    fn a_source_drop_does_not_invalidate_the_route() {
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let epoch = stereo.lock().unwrap().route_epoch;
        {
            let mut a = source(&stereo, &mono);
            a.next();
        }
        assert_eq!(
            stereo.lock().unwrap().route_epoch,
            epoch,
            "a source drop advanced the route epoch"
        );
        let mut b = source(&stereo, &mono);
        b.next();
        assert_eq!(
            stereo.lock().unwrap().channels,
            2,
            "a later source in the same session could not claim"
        );
    }
}

// ---------------------------------------------------------------------------
// Presentation discontinuities, the real-time snap, and the waterfall policy
// ---------------------------------------------------------------------------

/// The presentation state lives in two objects and only one of them was ever
/// reset. These drive the window, and check the things the *renderer* reads —
/// peak values it would draw, the texture it would paint — not only the
/// analyser fields behind them.
#[cfg(test)]
mod presentation_tests {
    use super::*;

    const BARS: usize = 8;
    const FFT: usize = 1024;
    const RATE: f64 = 180.0;

    fn window() -> SpectrumWindow {
        let mut w = SpectrumWindow::new();
        w.analyzer.sample_rate = 48_000;
        w.analyzer.fft_size = FFT;
        w.analyzer.bar_count = BARS;
        w.bar_count = BARS;
        w.analyzer.rebuild_fft();
        w.analyzer.magnitudes = vec![0.0; BARS];
        w.analyzer.smoothed = vec![0.0; BARS];
        w.analyzer.peak_input = vec![0.0; BARS];
        w.peak_config.enabled = true;
        w.style = VizStyle::Waterfall;
        w
    }

    fn cache(n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|f| {
                (0..BARS)
                    .map(|b| (((f * 7 + b * 13) % 97) as f32) / 97.0)
                    .collect()
            })
            .collect()
    }

    fn tone(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / 48_000.0).sin() * 0.5)
            .collect()
    }

    /// Put the window into a state where the renderer has plenty to draw:
    /// peaks the peak-hold pass would paint, and a waterfall ring with an
    /// upload watermark that says it is already on the GPU.
    fn fill_presentation(w: &mut SpectrumWindow) {
        w.analyzer.peak_input = vec![0.9; BARS];
        w.update_peaks(0.0);
        assert!(
            w.peak_vals.iter().any(|&v| v > 0.5),
            "setup: the renderer should have peaks to draw"
        );
        for _ in 0..40 {
            w.analyzer.waterfall_enabled = true;
            w.analyzer.push_waterfall_for_test(vec![0.5; BARS]);
        }
        // Stand in for a completed GPU upload without needing a context.
        w.waterfall_tex_w = BARS;
        w.waterfall_head = 7;
        w.waterfall_uploaded_seq = w.analyzer.waterfall_seq;
        assert!(!w.analyzer.waterfall.is_empty(), "setup: rows to draw");
    }

    fn assert_nothing_to_render(w: &SpectrumWindow, what: &str) {
        assert!(
            w.peak_vals.iter().all(|&v| v == 0.0),
            "{what}: peak markers from the previous producer are still drawable"
        );
        assert!(
            w.peak_alphas.iter().all(|&a| a == 1.0),
            "{what}: peak fade state survived"
        );
        assert!(
            w.analyzer.waterfall.is_empty(),
            "{what}: waterfall rows from the previous producer survived"
        );
        assert!(
            w.waterfall_texture.is_none(),
            "{what}: the uploaded waterfall texture is still drawable"
        );
        assert_eq!(w.waterfall_head, 0, "{what}: the ring head survived");
        assert_eq!(
            w.waterfall_uploaded_seq, w.analyzer.waterfall_seq,
            "{what}: the upload watermark still claims rows are on the GPU"
        );
        assert!(
            w.analyzer.magnitudes.iter().all(|&m| m == 0.0),
            "{what}: the spectrum survived"
        );
    }

    /// Stopping must leave the renderer with nothing to paint. `analyzer.reset`
    /// alone left peak markers and the GPU ring alive.
    #[test]
    fn stopping_leaves_nothing_for_the_renderer() {
        let mut w = window();
        fill_presentation(&mut w);
        w.on_stop();
        assert_nothing_to_render(&w, "on_stop");
    }

    /// Track A's presentation must not survive into track B.
    #[test]
    fn a_new_track_keeps_none_of_the_previous_ones_presentation() {
        let mut w = window();
        w.on_play(Path::new("C:/music/a.flac"), 48_000);
        fill_presentation(&mut w);
        w.on_play(Path::new("C:/music/b.flac"), 48_000);
        assert_nothing_to_render(&w, "on_play");
    }

    /// And a seek is the same discontinuity.
    #[test]
    fn a_seek_keeps_none_of_the_previous_positions_presentation() {
        let mut w = window();
        w.mode = SpectrumMode::RealTime;
        fill_presentation(&mut w);
        w.on_seek(12.0);
        assert_nothing_to_render(&w, "on_seek");
    }

    /// A cache arriving while the live-FFT fallback has been running is a
    /// producer change: the fallback's peaks and rows must not survive it, and
    /// the cache itself must not be destroyed by the reset it triggers.
    #[test]
    fn a_cache_handoff_resets_the_fallback_presentation() {
        let mut w = window();
        w.mode = SpectrumMode::PreProcess;
        // No cache yet: tick_pre falls back to the live FFT.
        {
            let mut b = w.sample_buf.lock().unwrap();
            b.extend_from_slice(&tone(1000.0, FFT * 2));
        }
        w.last_fft_time = None;
        w.tick(0.0, true);
        fill_presentation(&mut w);

        // The worker finishes.
        let frames = cache(2048);
        w.analyzer.set_pre_frames(frames.clone(), RATE);
        w.last_fft_time = None;
        w.analyzer.smoothing = 0.0;
        w.tick(300.0 / RATE, true);

        assert!(
            !w.analyzer.pre_frames.is_empty(),
            "the reset destroyed the cache that caused it"
        );
        assert_eq!(
            w.analyzer.magnitudes, frames[300],
            "the first cached tick must snap to the row the clock points at"
        );
        assert!(
            w.peak_vals
                .iter()
                .all(|&v| v <= w.analyzer.peak_input.iter().cloned().fold(0.0, f32::max)),
            "a fallback peak outlived the handoff"
        );
    }

    /// The same handoff while paused, where nothing is calling `tick_pre` and
    /// the only signal is the revision counter.
    #[test]
    fn a_cache_arriving_while_paused_resets_the_presentation() {
        let mut w = window();
        w.mode = SpectrumMode::PreProcess;
        fill_presentation(&mut w);
        w.analyzer.set_pre_frames(cache(1024), RATE);
        w.tick(5.0, false);
        assert_nothing_to_render(&w, "a paused cache handoff");
        assert!(!w.analyzer.pre_frames.is_empty(), "the cache was destroyed");
    }

    /// Dropping a cache is a producer change too.
    #[test]
    fn clearing_the_cache_resets_the_presentation() {
        let mut w = window();
        w.mode = SpectrumMode::PreProcess;
        w.analyzer.set_pre_frames(cache(1024), RATE);
        w.tick(0.0, false);
        fill_presentation(&mut w);
        w.analyzer.clear_pre_frames();
        w.tick(0.0, false);
        assert_nothing_to_render(&w, "clearing the cache");
    }

    /// An empty ring means nothing to draw. Returning early left the previous
    /// texture in place and `draw_waterfall` went on painting it.
    #[test]
    fn an_empty_waterfall_discards_the_texture_rather_than_leaving_it_drawable() {
        let mut w = window();
        fill_presentation(&mut w);
        // Model an upload having happened, then the ring being emptied without
        // the texture being told.
        w.waterfall_uploaded_seq = 0;
        w.analyzer.waterfall.clear();
        let ctx = egui::Context::default();
        let pal = Palette::new(w.palette_kind, w.palette_accent);
        let _ = ctx.run(Default::default(), |_| {});
        w.update_waterfall_texture(&ctx, &pal);
        assert!(
            w.waterfall_texture.is_none(),
            "an empty ring left a drawable texture behind"
        );
        assert_eq!(w.waterfall_head, 0);
        assert_eq!(w.waterfall_uploaded_seq, w.analyzer.waterfall_seq);
    }

    /// Every production write to the cached matrix must go through the two
    /// operations that keep the cursor and the revision in step with it.
    ///
    /// Not the only proof — the behaviour above is — but the one that catches a
    /// *new* direct write, which no behavioural test can be written for in
    /// advance.
    #[test]
    fn nothing_writes_the_cached_matrix_directly() {
        let src = include_str!("spectrum.rs");
        // Split so the needles do not match themselves: this test is inside the
        // file it scans, and a literal here would count as an occurrence.
        let assign = concat!("pre_frames", " = ");
        let clear = concat!("pre_frames", ".clear()");
        let assigns = src.matches(assign).count();
        let clears = src.matches(clear).count();
        assert_eq!(
            assigns, 1,
            "`pre_frames` is assigned {assigns} times; only `set_pre_frames` may"
        );
        assert_eq!(
            clears, 1,
            "`pre_frames` is cleared {clears} times; only `clear_pre_frames` may"
        );
    }
}

/// The first frame after a producer change has to *be* the audio, not fade up
/// towards it.
#[cfg(test)]
mod realtime_snap_tests {
    use super::*;

    const BARS: usize = 64;
    const FFT: usize = 1024;
    const RATE: f64 = 180.0;

    fn window() -> SpectrumWindow {
        let mut w = SpectrumWindow::new();
        w.analyzer.sample_rate = 48_000;
        w.analyzer.fft_size = FFT;
        w.analyzer.bar_count = BARS;
        w.bar_count = BARS;
        w.analyzer.min_freq = 20.0;
        w.analyzer.max_freq = 20_000.0;
        w.analyzer.rebuild_fft();
        w.analyzer.magnitudes = vec![0.0; BARS];
        w.analyzer.smoothed = vec![0.0; BARS];
        w.analyzer.peak_input = vec![0.0; BARS];
        w
    }

    fn tone(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / 48_000.0).sin() * 0.5)
            .collect()
    }

    fn cache(n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|f| {
                (0..BARS)
                    .map(|b| (((f * 7 + b * 13) % 97) as f32) / 97.0)
                    .collect()
            })
            .collect()
    }

    /// Entering Real-time with heavy smoothing and nonzero prior state: the
    /// first computed frame must equal the raw transform, and the second must
    /// resume the EMA.
    ///
    /// Zeroing the smoother is not enough — the elapsed time is short and the
    /// retention is high, so the first frame would come out at a few percent of
    /// the real value and the display would fade up from silence.
    #[test]
    fn the_first_realtime_frame_after_a_mode_change_is_a_snap() {
        let mut w = window();
        w.analyzer.smoothing = 0.97;
        w.mode = SpectrumMode::PreProcess;
        w.analyzer.set_pre_frames(cache(4096), RATE);
        // Real prior state from the other producer.
        w.last_fft_time = None;
        w.tick(1.0, true);
        assert!(
            w.analyzer.magnitudes.iter().any(|&m| m > 0.0),
            "setup: Pre-process should have left something on screen"
        );

        {
            let mut b = w.sample_buf.lock().unwrap();
            b.extend_from_slice(&tone(1000.0, FFT * 2));
        }
        // The unsmoothed answer, from a separate analyser on the same input.
        let want = {
            let mut probe = window();
            probe.analyzer.smoothing = 0.0;
            {
                let mut b = probe.sample_buf.lock().unwrap();
                b.extend_from_slice(&tone(1000.0, FFT * 2));
            }
            probe.analyzer.process_realtime(1.0 / 60.0);
            probe.analyzer.magnitudes.clone()
        };

        w.mode = SpectrumMode::RealTime;
        // The mode change is noticed before the cadence throttle, so this tick
        // arms the snap without computing a frame. The frame is then driven
        // with a *recent* interval — 1/60 s, where alpha at 0.97 retention is
        // 0.97 and a display without the snap would come out at 3% of the real
        // value. Clearing `last_fft_time` instead would pass a `dt` of
        // f64::MAX, which resolves to alpha 0 on its own and would let this
        // test succeed against no snap at all.
        w.tick(1.0 + 1.0 / 60.0, true);
        assert!(
            timing::alpha_for_dt(0.97, 1.0 / 60.0) > 0.9,
            "the interval used below must be one where smoothing would show"
        );
        w.analyzer.process_realtime(1.0 / 60.0);

        for (i, (&got, &wanted)) in w.analyzer.magnitudes.iter().zip(want.iter()).enumerate() {
            assert!(
                (got - wanted).abs() < 1e-6,
                "bar {i}: first Real-time frame {got} is not the transform {wanted}"
            );
        }

        // And the second frame resumes smoothing rather than snapping again.
        let first = w.analyzer.magnitudes.clone();
        {
            let mut b = w.sample_buf.lock().unwrap();
            b.clear();
            b.extend_from_slice(&tone(8000.0, FFT * 2));
        }
        w.analyzer.process_realtime(1.0 / 60.0);
        let second = w.analyzer.magnitudes.clone();
        assert_ne!(second, first, "the second frame did not update at all");
        // At 0.97 retention the second frame must still be dominated by the
        // first, which is what "the EMA resumed" means.
        let moved: f32 = second
            .iter()
            .zip(first.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let span: f32 = first.iter().cloned().fold(0.0, f32::max);
        assert!(
            moved < span * BARS as f32 * 0.2,
            "the second frame snapped too: moved {moved} against span {span}"
        );
    }

    /// A snap that arrives before there is enough PCM must not be spent on the
    /// frame that never happened.
    #[test]
    fn the_snap_survives_a_frame_that_could_not_be_computed() {
        let mut w = window();
        w.analyzer.smoothing = 0.97;
        w.analyzer.snap_next_realtime_frame();
        // Not enough history: process_realtime returns before transforming.
        {
            let mut b = w.sample_buf.lock().unwrap();
            b.extend_from_slice(&tone(1000.0, FFT / 4));
        }
        w.analyzer.process_realtime(1.0 / 60.0);
        assert!(
            w.analyzer.magnitudes.iter().all(|&m| m == 0.0),
            "setup: nothing should have been computed"
        );

        // Now there is enough, and this frame is the one that snaps.
        {
            let mut b = w.sample_buf.lock().unwrap();
            b.clear();
            b.extend_from_slice(&tone(1000.0, FFT * 2));
        }
        w.analyzer.process_realtime(1.0 / 60.0);
        let mut probe = window();
        probe.analyzer.smoothing = 0.0;
        {
            let mut b = probe.sample_buf.lock().unwrap();
            b.extend_from_slice(&tone(1000.0, FFT * 2));
        }
        probe.analyzer.process_realtime(1.0 / 60.0);
        assert_eq!(
            w.analyzer.magnitudes, probe.analyzer.magnitudes,
            "the snap was spent on a frame that was never computed"
        );
    }

    /// The reverse direction: entering Pre-process shows the row the clock
    /// points at, not a fade towards it.
    #[test]
    fn the_first_preprocess_frame_after_a_mode_change_is_a_snap() {
        let mut w = window();
        w.analyzer.smoothing = 0.97;
        w.mode = SpectrumMode::RealTime;
        let frames = cache(4096);
        w.analyzer.set_pre_frames(frames.clone(), RATE);
        {
            let mut b = w.sample_buf.lock().unwrap();
            b.extend_from_slice(&tone(1000.0, FFT * 2));
        }
        w.last_fft_time = None;
        w.tick(1.0, true);
        assert!(w.analyzer.magnitudes.iter().any(|&m| m > 0.0), "setup");

        w.mode = SpectrumMode::PreProcess;
        w.last_fft_time = None;
        w.tick(600.0 / RATE, true);
        assert_eq!(
            w.analyzer.magnitudes, frames[600],
            "the first Pre-process frame faded up instead of snapping"
        );
    }
}

// ---------------------------------------------------------------------------
// What the smoothing controls actually do, in milliseconds
// ---------------------------------------------------------------------------

/// The acceptance failure was arithmetic. These pin the arithmetic.
#[cfg(test)]
mod smoothing_response_tests {
    use super::*;

    /// The number the previous phase shipped, and the lag it produced.
    #[test]
    fn the_value_that_failed_acceptance_is_174_ms() {
        let ms = timing::step_response_95_secs(0.75) * 1000.0;
        assert!(
            (ms - 173.6).abs() < 0.5,
            "0.75 should be ~174 ms to 95%, got {ms:.1}"
        );
        // And what it becomes per row at the owner's analysis rate.
        let alpha = timing::alpha_for_dt(0.75, 1.0 / 180.0);
        assert!((alpha - 0.90856).abs() < 1e-4, "alpha was {alpha}");
    }

    /// The default, and the release it is calibrated to.
    #[test]
    fn the_default_reproduces_the_accepted_response() {
        let ms = timing::step_response_95_secs(timing::DEFAULT_PRE_SMOOTHING) * 1000.0;
        assert!(
            (ms - 24.0).abs() < 0.5,
            "default should be ~24 ms, got {ms:.1}"
        );
        // v1.4.5 hard-coded alpha 0.5 per tick and ran at ~180 Hz. This
        // produces exactly that alpha per row at that rate.
        let alpha = timing::alpha_for_dt(timing::DEFAULT_PRE_SMOOTHING, 1.0 / 180.0);
        assert!((alpha - 0.5).abs() < 1e-6, "alpha at 180 Hz was {alpha}");
    }

    /// And it holds at every source rate, which the hard-coded version could
    /// not: there the response was whatever the display happened to be doing.
    ///
    /// Two figures, because they are two different things and only one of them
    /// is the filter's.
    ///
    /// The *filter* reaches 95% in 24 ms of source time at every rate — that is
    /// what normalising by the interval buys, and it is the property under
    /// test. The *observable* step is then rounded up to the next row, because
    /// nothing can be shown between rows: a 46.875 Hz analysis has rows 21.3 ms
    /// apart, so it crosses 95% on its second row at 42.7 ms. That is the
    /// analysis rate, not the smoothing, and no setting can improve it.
    #[test]
    fn the_default_reaches_95_percent_within_30_ms_at_every_source_rate() {
        for rate in [46.875f64, 60.0, 144.0, 180.0, 240.0] {
            let alpha = timing::alpha_for_dt(timing::DEFAULT_PRE_SMOOTHING, 1.0 / rate);
            // Walk the actual recurrence rather than trusting the closed form.
            let mut v = 0.0f64;
            let mut rows = 0usize;
            while v < 0.95 && rows < 100_000 {
                v = v * alpha as f64 + 1.0 * (1.0 - alpha as f64);
                rows += 1;
            }
            let row_period_ms = 1000.0 / rate;
            let overshoot = (rows as f64 * row_period_ms - 24.0)
                .max(0.0)
                .min(row_period_ms);
            let filter_ms = rows as f64 * row_period_ms - overshoot;
            assert!(
                filter_ms <= 30.0,
                "{rate} Hz: the filter itself took {filter_ms:.1} ms to reach 95%"
            );
            // The quantised figure, asserted rather than glossed over.
            let observed_ms = rows as f64 * row_period_ms;
            assert!(
                observed_ms <= 30.0 + row_period_ms,
                "{rate} Hz: {observed_ms:.1} ms is more than one row past 30 ms"
            );
            assert_eq!(
                rows,
                (24.0f64 / row_period_ms).ceil() as usize,
                "{rate} Hz: {rows} rows is not the 24 ms response rounded up to a row"
            );
        }
    }

    /// The closed form, which is what the tooltip quotes, at every rate.
    #[test]
    fn the_quoted_response_is_rate_independent() {
        let quoted = timing::step_response_95_secs(timing::DEFAULT_PRE_SMOOTHING);
        assert!((quoted * 1000.0 - 24.0).abs() < 0.5);
        for rate in [46.875f64, 60.0, 144.0, 180.0, 240.0] {
            let alpha = timing::alpha_for_dt(timing::DEFAULT_PRE_SMOOTHING, 1.0 / rate) as f64;
            // Retention compounded over the quoted time must be 5%, whatever
            // the rate: alpha^(rate*t) = 0.05.
            let residue = alpha.powf(rate * quoted);
            assert!(
                (residue - 0.05).abs() < 1e-6,
                "{rate} Hz: {residue} of the step remained after the quoted {quoted:.4}s"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Two settings, two meanings, migrated honestly
// ---------------------------------------------------------------------------

#[cfg(test)]
mod smoothing_settings_tests {
    use super::*;

    /// A settings file written before the split. The pre-process path ignored
    /// `smoothing` entirely then, so there is no value to inherit — inheriting
    /// 0.75 would hand the user 174 ms they never chose and never saw.
    #[test]
    fn a_legacy_settings_file_takes_the_documented_pre_default() {
        let legacy = r#"{"smoothing":0.75}"#;
        let s: SpectrumSettings = serde_json::from_str(legacy).expect("legacy settings");
        assert_eq!(s.pre_smoothing, None, "the field must be absent, not zero");

        let mut w = SpectrumWindow::new();
        w.apply_settings(&s);
        assert_eq!(
            w.smoothing, 0.75,
            "the Real-time setting the file did carry must be preserved"
        );
        assert_eq!(
            w.pre_smoothing,
            timing::DEFAULT_PRE_SMOOTHING,
            "a missing pre-process value must take the default, not inherit 0.75"
        );
    }

    /// Zero is a setting, not an absence — which is why the field is `Option`
    /// rather than `#[serde(default)]` on a bare `f32`.
    #[test]
    fn a_saved_zero_is_kept_as_off() {
        let saved = r#"{"smoothing":0.5,"pre_smoothing":0.0}"#;
        let s: SpectrumSettings = serde_json::from_str(saved).expect("settings");
        assert_eq!(s.pre_smoothing, Some(0.0));
        let mut w = SpectrumWindow::new();
        w.apply_settings(&s);
        assert_eq!(w.pre_smoothing, 0.0, "an explicit Off was upgraded away");
    }

    #[test]
    fn the_two_values_round_trip_independently() {
        let mut w = SpectrumWindow::new();
        w.smoothing = 0.61;
        w.pre_smoothing = 0.07;
        let json = serde_json::to_string(&w.snapshot()).unwrap();
        let back: SpectrumSettings = serde_json::from_str(&json).unwrap();

        let mut other = SpectrumWindow::new();
        other.apply_settings(&back);
        assert!((other.smoothing - 0.61).abs() < 1e-6);
        assert!((other.pre_smoothing - 0.07).abs() < 1e-6);

        // Changing one must not move the other.
        other.pre_smoothing = 0.5;
        let json2 = serde_json::to_string(&other.snapshot()).unwrap();
        let back2: SpectrumSettings = serde_json::from_str(&json2).unwrap();
        let mut third = SpectrumWindow::new();
        third.apply_settings(&back2);
        assert!(
            (third.smoothing - 0.61).abs() < 1e-6,
            "Real-time moved with Pre-process"
        );
        assert!((third.pre_smoothing - 0.5).abs() < 1e-6);
    }

    /// Neither value is part of the analysis, so neither can invalidate one.
    #[test]
    fn neither_smoothing_value_touches_the_cache_identity() {
        let path = PathBuf::from("C:/music/track.flac");
        let cfg = aslt::AsltPreset::Standard.config();
        let key = || {
            cache_path_for(
                &path,
                1024,
                8192,
                16,
                0.875,
                &WindowFn::Hann,
                20.0,
                24_000.0,
                &BarMappingMode::Superlet,
                &InterpolationMode::None,
                176_400,
                &cfg,
                180.0,
            )
        };
        let before = key();

        let mut w = SpectrumWindow::new();
        w.analyzer.set_pre_frames(vec![vec![0.5; 8]; 32], 180.0);
        w.mode = SpectrumMode::PreProcess;
        w.needs_reanalysis = false;
        for v in [0.0f32, 0.125, 0.75, 0.97] {
            w.smoothing = v;
            w.pre_smoothing = v;
            w.sync_params();
            assert_eq!(key(), before, "smoothing {v} moved the cache key");
            assert!(!w.needs_reanalysis, "smoothing {v} asked for reanalysis");
            assert!(
                !w.analyzer.pre_frames.is_empty(),
                "smoothing {v} dropped the cache"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The Real-time recurrence, unchanged from the accepted release
// ---------------------------------------------------------------------------

#[cfg(test)]
mod realtime_recurrence_tests {
    use super::*;

    const BARS: usize = 32;
    const FFT: usize = 1024;

    fn tone(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / 48_000.0).sin() * 0.5)
            .collect()
    }

    fn analyzer(smoothing: f32) -> SpectrumAnalyzer {
        let mut a = SpectrumAnalyzer::new(new_sample_buf());
        a.sample_rate = 48_000;
        a.fft_size = FFT;
        a.bar_count = BARS;
        a.min_freq = 20.0;
        a.max_freq = 20_000.0;
        a.smoothing = smoothing;
        a.rebuild_fft();
        a.magnitudes = vec![0.0; BARS];
        a.smoothed = vec![0.0; BARS];
        a.peak_input = vec![0.0; BARS];
        {
            let mut b = a.sample_buf.lock().unwrap();
            b.extend_from_slice(&tone(1000.0, FFT * 2));
        }
        a
    }

    /// v1.4.5 applied the setting once per accepted tick, whatever the interval
    /// was. That is the response the owner's ears are calibrated to, and it is
    /// what ships again — normalising it to a reference rate made Real-time
    /// about three times slower on a fast display, which was never asked for.
    #[test]
    fn realtime_applies_the_setting_once_per_accepted_tick() {
        const S: f32 = 0.8;
        for ticks in [60usize, 144, 180, 240] {
            let mut a = analyzer(S);
            // Spend the snap on a first frame, then change the input. A steady
            // input is useless here: `old*a + frame*(1-a)` with `old == frame`
            // returns `frame` for *every* alpha, so a test that never changes
            // the signal cannot see the smoothing at all.
            a.process_realtime(1.0 / ticks as f64);
            let start = a.magnitudes.clone();

            {
                let mut b = a.sample_buf.lock().unwrap();
                b.clear();
                b.extend_from_slice(&tone(6000.0, FFT * 2));
            }
            // The frame the recurrence is now converging on, measured
            // independently so the expectation is not taken from the thing
            // under test.
            let target = {
                let mut probe = analyzer(0.0);
                {
                    let mut b = probe.sample_buf.lock().unwrap();
                    b.clear();
                    b.extend_from_slice(&tone(6000.0, FFT * 2));
                }
                probe.process_realtime(1.0 / ticks as f64);
                probe.magnitudes.clone()
            };
            assert_ne!(start, target, "setup: the input must actually change");

            for _ in 0..4 {
                a.process_realtime(1.0 / ticks as f64);
            }

            // Four applications of `old*S + frame*(1-S)`, which is the closed
            // form of the v1.4.5 recurrence — and, crucially, has S in it.
            for (i, ((&got, &s0), &t)) in a
                .magnitudes
                .iter()
                .zip(start.iter())
                .zip(target.iter())
                .enumerate()
            {
                let mut want = s0;
                for _ in 0..4 {
                    want = want * S + t * (1.0 - S);
                }
                assert!(
                    (got - want).abs() < 1e-5,
                    "{ticks} ticks, bar {i}: {got} vs {want}"
                );
            }
        }
    }

    /// The interval must not enter into it. A tick is a tick.
    #[test]
    fn the_realtime_result_does_not_depend_on_the_interval() {
        let mut fast = analyzer(0.8);
        let mut slow = analyzer(0.8);
        // Snap both, then converge both on a different frame, so the interval
        // has something to be wrong about.
        fast.process_realtime(1.0 / 240.0);
        slow.process_realtime(1.0 / 60.0);
        for a in [&mut fast, &mut slow] {
            let mut b = a.sample_buf.lock().unwrap();
            b.clear();
            b.extend_from_slice(&tone(6000.0, FFT * 2));
        }
        for _ in 0..6 {
            fast.process_realtime(1.0 / 240.0);
            slow.process_realtime(1.0 / 60.0);
        }
        assert_eq!(
            fast.magnitudes, slow.magnitudes,
            "Real-time smoothing became a function of the tick interval again"
        );
    }
}

// ---------------------------------------------------------------------------
// One lease, both buffers
// ---------------------------------------------------------------------------

/// The route epoch protected the stereo buffer and left the mono one open, so
/// every teardown left the Mix — the series most people are actually watching —
/// still being written by a source whose session was over. These drive the
/// orderings that produce that, and check *both* buffers every time.
#[cfg(test)]
mod live_pcm_lease_tests {
    use super::stereo_tap_tests::{Pattern, interleaved};
    use super::*;

    const FRAMES: usize = 8192;

    fn source(stereo: &StereoBuf, mono: &SampleBuf) -> SpectrumSource<Pattern> {
        SpectrumSource::new(
            Pattern {
                data: interleaved(FRAMES),
                at: 0,
                channels: 2,
            },
            Arc::clone(mono),
            Arc::clone(stereo),
        )
    }

    /// Distinctive contents in both buffers, so "unchanged" is a real assertion.
    fn sentinels(stereo: &StereoBuf, mono: &SampleBuf) -> (Vec<[f32; 2]>, Vec<f32>) {
        let pairs: Vec<[f32; 2]> = (0..64)
            .map(|i| [9000.0 + i as f32, -(9000.0 + i as f32)])
            .collect();
        let flat: Vec<f32> = (0..64).map(|i| 5000.0 + i as f32).collect();
        stereo.lock().unwrap().frames.extend_from_slice(&pairs);
        mono.lock().unwrap().extend_from_slice(&flat);
        (pairs, flat)
    }

    fn assert_untouched(
        stereo: &StereoBuf,
        mono: &SampleBuf,
        pairs: &[[f32; 2]],
        flat: &[f32],
        what: &str,
    ) {
        assert_eq!(
            stereo.lock().unwrap().frames,
            pairs,
            "{what}: the live route's stereo frames were disturbed"
        );
        assert_eq!(
            *mono.lock().unwrap(),
            flat,
            "{what}: the live route's mono PCM was disturbed"
        );
    }

    /// Forced schedule 1. Constructed, never pulled, route ends, a successor
    /// installs sentinels in both buffers — then the mixer finally enters the
    /// old source. Neither buffer may move.
    #[test]
    fn a_never_pulled_source_cannot_write_either_buffer_after_its_route_ended() {
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let mut old = source(&stereo, &mono);

        invalidate_live_pcm_route(&stereo, &mono);
        let live = begin_stereo_stream(&stereo, 2);
        let (pairs, flat) = sentinels(&stereo, &mono);

        for _ in 0..(BATCH_SIZE * 8) {
            old.next();
        }

        assert_eq!(stereo.lock().unwrap().generation, live);
        assert_untouched(&stereo, &mono, &pairs, &flat, "a stale never-pulled source");
    }

    /// Forced schedule 2. An owner with a nearly-full local batch, parked across
    /// the invalidation and the successor's claim. Its final sample completes
    /// the batch and triggers a flush — which must reach neither buffer.
    #[test]
    fn an_owner_parked_across_teardown_publishes_to_neither_buffer() {
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let mut old = source(&stereo, &mono);

        // Own the lease, then fill the local batch to one sample short of a
        // flush. Two interleaved samples make one frame, so this is exact.
        for _ in 0..(BATCH_SIZE * 2 - 2) {
            old.next();
        }
        assert!(
            !mono.lock().unwrap().is_empty() || true,
            "setup only: the batch may or may not have flushed yet"
        );

        // The route ends and a successor takes over while the batch is held.
        invalidate_live_pcm_route(&stereo, &mono);
        let live = begin_stereo_stream(&stereo, 2);
        let (pairs, flat) = sentinels(&stereo, &mono);

        // The samples that complete the batch and fire the flush.
        old.next();
        old.next();

        assert_eq!(stereo.lock().unwrap().generation, live);
        assert_untouched(&stereo, &mono, &pairs, &flat, "a parked owner's last flush");
    }

    /// A source that has not claimed yet is not the live one, and must not
    /// publish mono while it waits. It used to.
    #[test]
    fn a_pending_source_publishes_nothing() {
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let mut src = source(&stereo, &mono);
        // Held for the whole run, so every claim attempt returns Busy.
        let _held = stereo.lock().unwrap();
        for _ in 0..(BATCH_SIZE * 8) {
            src.next();
        }
        assert!(
            mono.lock().unwrap().is_empty(),
            "a source that never claimed still fed the Mix"
        );
    }

    /// A source whose route ended gathers nothing, as well as publishing
    /// nothing.
    ///
    /// Publication is already gated on ownership, so a Disabled source that
    /// kept filling its batches would be invisible in the buffers — it would
    /// simply do the work of a live tap, on the audio thread, for the rest of
    /// the track. The batches are what has to be checked.
    #[test]
    fn a_disabled_source_gathers_nothing() {
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let mut src = source(&stereo, &mono);
        invalidate_live_pcm_route(&stereo, &mono);

        // The first pull is refused, permanently.
        src.next();
        // Deliberately *not* a whole number of batches. A count that lands on a
        // flush boundary leaves the batch empty however the source behaves, and
        // this assertion would hold against a source that gathered everything.
        for _ in 0..(BATCH_SIZE * 4 + 20) {
            src.next();
        }

        assert!(
            src.sample_batch.is_empty(),
            "a disabled source gathered {} mono frames it can never publish",
            src.sample_batch.len()
        );
        assert!(
            src.stereo_batch.is_empty(),
            "a disabled source gathered {} pairs it can never publish",
            src.stereo_batch.len()
        );
        assert!(mono.lock().unwrap().is_empty());
        assert!(stereo.lock().unwrap().frames.is_empty());
    }

    /// Forced schedule 3. The bit-perfect tap, superseded, must alter neither.
    #[test]
    fn a_superseded_bit_perfect_tap_writes_neither_buffer() {
        // Driven through the same publication helper the tap uses, with a lease
        // that is no longer current.
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let stale = begin_stereo_stream(&stereo, 2);
        invalidate_live_pcm_route(&stereo, &mono);
        begin_stereo_stream(&stereo, 2);
        let (pairs, flat) = sentinels(&stereo, &mono);

        let published = publish_live_pcm(
            &stereo,
            &mono,
            stale,
            &[0.25f32; 128],
            &[[0.25f32, -0.25]; 128],
        );
        assert!(!published, "a stale lease published");
        assert_untouched(
            &stereo,
            &mono,
            &pairs,
            &flat,
            "a superseded bit-perfect tap",
        );
    }

    /// Forced schedule 4. Teardown cannot interleave with a publication.
    ///
    /// The hook stands between the lease check and the first write, which is
    /// the only place an interleaving could happen. It proves the ownership
    /// guard is still held there — so a teardown, which needs that same guard,
    /// is either wholly before the publication or wholly after it.
    #[test]
    fn teardown_cannot_cross_a_publication() {
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        let lease = begin_stereo_stream(&stereo, 2);

        let blocked = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = Arc::clone(&blocked);
        let probe = Arc::clone(&stereo);
        let _hook = publish_hook::install(move || {
            // A teardown would need this lock. It cannot have it.
            assert!(
                probe.try_lock().is_err(),
                "the ownership guard was not held across publication"
            );
            seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });

        assert!(publish_live_pcm(
            &stereo,
            &mono,
            lease,
            &[1.0f32; 8],
            &[[1.0f32, -1.0]; 8]
        ));
        assert_eq!(
            blocked.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "the observation point was not reached"
        );
        assert_eq!(mono.lock().unwrap().len(), 8);
        assert_eq!(stereo.lock().unwrap().frames.len(), 8);
    }

    /// Forced schedule 5. Every way a route ends clears both buffers.
    #[test]
    fn every_teardown_clears_both_buffers() {
        let mono = new_sample_buf();
        let stereo = new_stereo_buf();
        begin_stereo_stream(&stereo, 2);
        sentinels(&stereo, &mono);

        invalidate_live_pcm_route(&stereo, &mono);

        let g = stereo.lock().unwrap();
        assert!(g.frames.is_empty(), "stereo frames survived teardown");
        assert_eq!(g.channels, channels::NO_LIVE_TAP);
        drop(g);
        assert!(
            mono.lock().unwrap().is_empty(),
            "mono PCM survived teardown — the bars would keep drawing it"
        );
    }

    /// Forced schedule 8. Allocation-free through the first flush, later
    /// flushes, a Busy claim, a Stale claim, and a revoked lease.
    #[test]
    fn no_flush_state_allocates() {
        // First and later flushes, from an untouched source.
        {
            let mono = new_sample_buf();
            let stereo = new_stereo_buf();
            let mut src = source(&stereo, &mono);
            let before = crate::alloc_probe::arm();
            for _ in 0..(BATCH_SIZE * 8) {
                src.next();
            }
            let n = crate::alloc_probe::disarm(before);
            assert_eq!(n, 0, "first and steady-state flushes allocated {n} time(s)");
            assert!(!mono.lock().unwrap().is_empty(), "and did publish");
        }
        // Busy: the lock is held for the whole run, so every claim is refused.
        {
            let mono = new_sample_buf();
            let stereo = new_stereo_buf();
            let mut src = source(&stereo, &mono);
            let _held = stereo.lock().unwrap();
            let before = crate::alloc_probe::arm();
            for _ in 0..(BATCH_SIZE * 8) {
                src.next();
            }
            let n = crate::alloc_probe::disarm(before);
            assert_eq!(n, 0, "a Busy claim allocated {n} time(s)");
        }
        // Stale: the route ended before the first pull.
        {
            let mono = new_sample_buf();
            let stereo = new_stereo_buf();
            let mut src = source(&stereo, &mono);
            invalidate_live_pcm_route(&stereo, &mono);
            let before = crate::alloc_probe::arm();
            for _ in 0..(BATCH_SIZE * 8) {
                src.next();
            }
            let n = crate::alloc_probe::disarm(before);
            assert_eq!(n, 0, "a Stale claim allocated {n} time(s)");
        }
        // Revoked: owned, then superseded mid-stream.
        {
            let mono = new_sample_buf();
            let stereo = new_stereo_buf();
            let mut src = source(&stereo, &mono);
            for _ in 0..(BATCH_SIZE * 4) {
                src.next();
            }
            invalidate_live_pcm_route(&stereo, &mono);
            begin_stereo_stream(&stereo, 2);
            let before = crate::alloc_probe::arm();
            for _ in 0..(BATCH_SIZE * 8) {
                src.next();
            }
            let n = crate::alloc_probe::disarm(before);
            assert_eq!(n, 0, "a revoked lease allocated {n} time(s)");
        }
    }
}

// ---------------------------------------------------------------------------
// The snap reaches every series, and the reset reaches every drawable
// ---------------------------------------------------------------------------

#[cfg(test)]
mod frame_snap_tests {
    use super::*;

    const BARS: usize = 64;
    const FFT: usize = 1024;

    fn tone(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / 48_000.0).sin() * 0.5)
            .collect()
    }

    fn window(smoothing: f32) -> SpectrumWindow {
        let mut w = SpectrumWindow::new();
        w.mode = SpectrumMode::RealTime;
        w.style = VizStyle::Bars;
        w.smoothing = smoothing;
        w.analyzer.smoothing = smoothing;
        w.analyzer.sample_rate = 48_000;
        w.analyzer.fft_size = FFT;
        w.analyzer.bar_count = BARS;
        w.bar_count = BARS;
        w.analyzer.min_freq = 20.0;
        w.analyzer.max_freq = 20_000.0;
        w.analyzer.rebuild_fft();
        w.analyzer.magnitudes = vec![0.0; BARS];
        w.analyzer.smoothed = vec![0.0; BARS];
        w.analyzer.peak_input = vec![0.0; BARS];
        w
    }

    fn feed(w: &SpectrumWindow, freq: f32) {
        let mut b = w.sample_buf.lock().unwrap();
        b.clear();
        b.extend_from_slice(&tone(freq, FFT * 2));
        drop(b);
        begin_stereo_stream(&w.stereo_buf, 2);
        let l = tone(freq, FFT * 2);
        let r = tone(freq * 2.0, FFT * 2);
        let mut g = w.stereo_buf.lock().unwrap();
        for i in 0..(FFT * 2) {
            g.frames.push([l[i], r[i]]);
        }
    }

    /// Every series computed from one frame must share that frame's decision.
    ///
    /// The snap used to be consumed by the Mix alone, so after any reset the
    /// Left, Right and octave series each applied ordinary smoothing and crept
    /// up from zero behind a Mix that was already correct. With retention 0.97
    /// that is visible for well over a second.
    #[test]
    fn the_first_frame_snaps_on_every_series_not_only_the_mix() {
        let mut w = window(0.97);
        w.channel_view = channels::ChannelView::Split;
        w.style = VizStyle::Bars;
        feed(&w, 1000.0);
        // A genuinely recent interval — 20 ms against a 60 Hz throttle, so the
        // tick is accepted and `dt` is small. Clearing `last_fft_time` instead
        // would hand every consumer a `dt` of `f64::MAX`, which resolves to
        // alpha 0 on its own: the test would pass against a build with no snap
        // at all, which is exactly how this one first did.
        w.max_fps = 60.0;
        w.last_fft_time = Some(Instant::now() - Duration::from_millis(20));
        w.tick(0.0, true);

        // Mix: equal to the unsmoothed transform.
        let want_mix = {
            let mut probe = window(0.0);
            {
                let mut b = probe.sample_buf.lock().unwrap();
                b.extend_from_slice(&tone(1000.0, FFT * 2));
            }
            probe.analyzer.process_realtime(1.0 / 60.0);
            probe.analyzer.magnitudes.clone()
        };
        for (i, (&got, &wanted)) in w.analyzer.magnitudes.iter().zip(want_mix.iter()).enumerate()
        {
            assert!((got - wanted).abs() < 1e-6, "Mix bar {i}: {got} vs {wanted}");
        }

        // Left and Right: present, and at full scale rather than 3% of it.
        let frame = w.analyzer.channel_frame();
        let (l, r) = frame.pair().expect("a two-channel tap must yield a pair");
        let lpeak = l.iter().cloned().fold(0.0f32, f32::max);
        let rpeak = r.iter().cloned().fold(0.0f32, f32::max);
        let mixpeak = want_mix.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            lpeak > mixpeak * 0.5,
            "Left peaked at {lpeak} against a Mix of {mixpeak} — it faded up instead of snapping"
        );
        assert!(rpeak > mixpeak * 0.5, "Right peaked at {rpeak}");

        // Octave meters: same.
        assert!(
            !w.octave_bands.is_empty(),
            "the octave meters produced nothing"
        );
        let opeak = w.octave_bands.iter().map(|&(_, v)| v).fold(0.0f32, f32::max);
        assert!(
            opeak > 0.2,
            "the octave meters peaked at {opeak} — they crept up instead of snapping"
        );
    }

    /// And the second genuine frame resumes smoothing on all of them.
    #[test]
    fn the_second_frame_resumes_smoothing_on_every_series() {
        let mut w = window(0.97);
        w.channel_view = channels::ChannelView::Split;
        feed(&w, 1000.0);
        w.max_fps = 60.0;
        w.last_fft_time = Some(Instant::now() - Duration::from_millis(20));
        w.tick(0.0, true);

        let mix_first = w.analyzer.magnitudes.clone();
        let oct_first: Vec<f32> = w.octave_bands.iter().map(|&(_, v)| v).collect();
        let l_first = w.analyzer.bars_left.clone();

        // A very different frame. At 0.97 retention almost none of it should
        // arrive in one step.
        feed(&w, 8000.0);
        w.last_fft_time = Some(Instant::now() - Duration::from_millis(20));
        w.tick(1.0 / 60.0, true);

        let moved: f32 = w
            .analyzer
            .magnitudes
            .iter()
            .zip(mix_first.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let span: f32 = mix_first.iter().cloned().fold(0.0, f32::max).max(1e-6);
        assert!(
            moved < span * BARS as f32 * 0.25,
            "the Mix snapped a second time: moved {moved} against span {span}"
        );

        let oct_moved: f32 = w
            .octave_bands
            .iter()
            .map(|&(_, v)| v)
            .zip(oct_first.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let oct_span: f32 = oct_first.iter().cloned().fold(0.0, f32::max).max(1e-6);
        assert!(
            oct_moved < oct_span * oct_first.len() as f32 * 0.25,
            "the octave meters snapped a second time"
        );

        let l_moved: f32 = w
            .analyzer
            .bars_left
            .iter()
            .zip(l_first.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let l_span: f32 = l_first.iter().cloned().fold(0.0, f32::max).max(1e-6);
        assert!(
            l_moved < l_span * BARS as f32 * 0.25,
            "Left snapped a second time"
        );
    }

    /// An insufficient-PCM frame is not a frame, and must not spend the snap
    /// for the channel and octave series either.
    #[test]
    fn a_starved_frame_spends_no_series_snap() {
        let mut w = window(0.97);
        w.channel_view = channels::ChannelView::Split;
        {
            let mut b = w.sample_buf.lock().unwrap();
            b.extend_from_slice(&tone(1000.0, FFT / 4));
        }
        w.max_fps = 60.0;
        w.last_fft_time = Some(Instant::now() - Duration::from_millis(20));
        w.tick(0.0, true);
        assert!(
            w.analyzer.magnitudes.iter().all(|&m| m == 0.0),
            "setup: nothing should have been computed"
        );

        feed(&w, 1000.0);
        w.last_fft_time = Some(Instant::now() - Duration::from_millis(20));
        w.tick(1.0 / 60.0, true);
        let peak = w.analyzer.magnitudes.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            peak > 0.2,
            "the snap was spent on a frame that never happened"
        );
    }
}

/// Everything that can still be painted after a discontinuity.
#[cfg(test)]
mod presentation_completeness_tests {
    use super::*;

    const BARS: usize = 8;
    const RATE: f64 = 180.0;

    fn window() -> SpectrumWindow {
        let mut w = SpectrumWindow::new();
        w.analyzer.sample_rate = 48_000;
        w.analyzer.bar_count = BARS;
        w.bar_count = BARS;
        w.peak_config.enabled = true;
        w.analyzer.magnitudes = vec![0.0; BARS];
        w.analyzer.smoothed = vec![0.0; BARS];
        w.analyzer.peak_input = vec![0.0; BARS];
        w
    }

    /// A real texture, not `None`. The previous version of this check started
    /// from `None` and therefore passed against a build that never cleared one.
    fn real_texture(ctx: &egui::Context, name: &str) -> egui::TextureHandle {
        ctx.load_texture(
            name,
            egui::ColorImage::new([4, 4], Color32::from_rgb(200, 30, 30)),
            egui::TextureOptions::NEAREST,
        )
    }

    /// Seed every drawable with something that is not its default.
    fn fill(w: &mut SpectrumWindow, ctx: &egui::Context) {
        w.analyzer.magnitudes = vec![0.7; BARS];
        w.analyzer.smoothed = vec![0.7; BARS];
        w.analyzer.peak_input = vec![0.9; BARS];
        w.analyzer.last_fft_norms = vec![0.6; 512];
        w.analyzer.bars_left = vec![0.8; BARS];
        w.analyzer.bars_right = vec![0.4; BARS];
        w.update_peaks(0.0);
        w.analyzer.waterfall_enabled = true;
        for _ in 0..40 {
            w.analyzer.push_waterfall_for_test(vec![0.5; BARS]);
        }
        w.waterfall_texture = Some(real_texture(ctx, "moosik_waterfall_test"));
        w.waterfall_tex_w = BARS;
        w.waterfall_head = 7;
        w.waterfall_uploaded_seq = w.analyzer.waterfall_seq;
        w.spectrogram.pixels.fill(Color32::from_rgb(90, 10, 10));
        w.spectrogram.col_head = 11;
        w.spectrogram.dirty = false;
        w.spectrogram_texture = Some(real_texture(ctx, "moosik_spectrogram_test"));
        w.octave_bands = vec![(100.0, 0.9), (1000.0, 0.8)];
        w.octave_smoothed = vec![0.9, 0.8];
        w.phasescope_frames = vec![[0.5, -0.5]; 64];
        w.correlation = -0.9;
        w.momentary_lufs = -7.5;
        w.lufs_scratch = vec![0.3; 256];

        assert!(w.peak_vals.iter().any(|&v| v > 0.5), "setup: peaks to draw");
    }

    fn assert_clean(w: &SpectrumWindow, what: &str) {
        assert!(
            w.analyzer.magnitudes.iter().all(|&v| v == 0.0),
            "{what}: bars"
        );
        assert!(
            w.analyzer.smoothed.iter().all(|&v| v == 0.0),
            "{what}: smoother"
        );
        assert!(
            w.analyzer.peak_input.iter().all(|&v| v == 0.0),
            "{what}: peak input"
        );
        assert!(
            w.analyzer.last_fft_norms.is_empty(),
            "{what}: last_fft_norms would re-seed the spectrogram and meters"
        );
        assert!(w.analyzer.bars_left.is_empty(), "{what}: left bars");
        assert!(w.analyzer.bars_right.is_empty(), "{what}: right bars");
        assert!(w.peak_vals.iter().all(|&v| v == 0.0), "{what}: peak values");
        assert!(
            w.peak_hold_timers.iter().all(|&v| v == 0.0),
            "{what}: peak timers"
        );
        assert!(
            w.peak_velocities.iter().all(|&v| v == 0.0),
            "{what}: peak velocities"
        );
        assert!(w.peak_alphas.iter().all(|&v| v == 1.0), "{what}: peak alphas");
        assert!(w.analyzer.waterfall.is_empty(), "{what}: waterfall rows");
        assert!(
            w.waterfall_texture.is_none(),
            "{what}: waterfall texture is drawable"
        );
        assert_eq!(w.waterfall_head, 0, "{what}: waterfall head");
        assert_eq!(
            w.waterfall_uploaded_seq, w.analyzer.waterfall_seq,
            "{what}: waterfall upload watermark"
        );
        assert!(
            w.spectrogram.pixels.iter().all(|&p| p == Color32::BLACK),
            "{what}: spectrogram pixels are still the previous track's"
        );
        assert_eq!(w.spectrogram.col_head, 0, "{what}: spectrogram head");
        assert!(
            w.spectrogram_texture.is_none(),
            "{what}: spectrogram texture is drawable"
        );
        assert!(w.octave_bands.is_empty(), "{what}: octave bands");
        assert!(w.octave_smoothed.is_empty(), "{what}: octave smoother");
        assert!(w.phasescope_frames.is_empty(), "{what}: phasescope frames");
        assert_eq!(w.correlation, 1.0, "{what}: correlation");
        assert_eq!(w.momentary_lufs, f32::NEG_INFINITY, "{what}: LUFS");
        assert!(w.lufs_scratch.is_empty(), "{what}: LUFS scratch");
    }

    #[test]
    fn stop_clears_every_drawable() {
        let ctx = egui::Context::default();
        let mut w = window();
        fill(&mut w, &ctx);
        w.on_stop();
        assert_clean(&w, "on_stop");
    }

    #[test]
    fn a_seek_clears_every_drawable_but_keeps_the_waveform() {
        let ctx = egui::Context::default();
        let mut w = window();
        w.mode = SpectrumMode::RealTime;
        w.waveform = Some(vec![0.5; 128]);
        fill(&mut w, &ctx);
        w.on_seek(12.0);
        assert_clean(&w, "on_seek");
        assert!(
            w.waveform.is_some(),
            "a seek discarded the full-track waveform, which it does not invalidate"
        );
    }

    #[test]
    fn a_new_track_clears_every_drawable() {
        let ctx = egui::Context::default();
        let mut w = window();
        fill(&mut w, &ctx);
        w.on_play(Path::new("C:/music/next.flac"), 48_000);
        assert_clean(&w, "on_play");
    }

    #[test]
    fn a_mode_switch_clears_every_drawable() {
        let ctx = egui::Context::default();
        let mut w = window();
        w.mode = SpectrumMode::RealTime;
        w.tick(0.0, false);
        fill(&mut w, &ctx);
        w.mode = SpectrumMode::PreProcess;
        w.tick(0.0, false);
        assert_clean(&w, "a mode switch");
    }

    #[test]
    fn a_cache_revision_change_clears_every_drawable() {
        let ctx = egui::Context::default();
        let mut w = window();
        w.mode = SpectrumMode::PreProcess;
        w.tick(0.0, false);
        fill(&mut w, &ctx);
        w.analyzer
            .set_pre_frames(vec![vec![0.25; BARS]; 512], RATE);
        w.tick(0.0, false);
        assert_clean(&w, "a cache handoff");
        assert!(!w.analyzer.pre_frames.is_empty(), "the new cache was destroyed");
    }
}

// ---------------------------------------------------------------------------
// The difference view
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The corner readouts share one row
// ---------------------------------------------------------------------------

/// LUFS, the channel legend and the frame rate all live along the top of the
/// plot. They were anchored independently by three pieces of code that each
/// assumed the row was empty, and at the default window size the LUFS number
/// ran straight through the `L`/`R` legend.
///
/// These drive the production layout and the production legend and assert the
/// drawn boxes are disjoint.
#[cfg(test)]
mod readout_row_tests {
    use super::*;

    fn plot(w: f32) -> Rect {
        let outer = Rect::from_min_size(Pos2::new(12.0, 8.0), egui::vec2(w, 460.0));
        Rect::from_min_max(
            Pos2::new(outer.left() + DB_MARGIN, outer.top()),
            Pos2::new(outer.right(), outer.bottom() - FREQ_MARGIN),
        )
    }

    fn capture(f: impl FnOnce(&egui::Painter)) -> Vec<egui::epaint::ClippedShape> {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(1600.0, 900.0))),
            ..Default::default()
        };
        let mut f = Some(f);
        let out = ctx.run(input, |ctx| {
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("readout_row_tests"),
            ));
            if let Some(f) = f.take() {
                f(&painter);
            }
        });
        out.shapes
    }

    /// Every text box drawn, in order.
    fn boxes(shapes: &[egui::epaint::ClippedShape]) -> Vec<(String, Rect)> {
        shapes
            .iter()
            .filter_map(|c| match &c.shape {
                egui::Shape::Text(t) => Some((
                    t.galley.text().to_string(),
                    Rect::from_min_size(t.pos, t.galley.size()),
                )),
                _ => None,
            })
            .collect()
    }

    /// Draw the whole row exactly as production does, and return what landed.
    ///
    /// Including the analysis badge's background, which is the part that
    /// actually collides — it is wider than its text and it is what was landing
    /// on the frame rate.
    fn draw_row(
        width: f32,
        lufs: Option<&str>,
        fps: &str,
        view: channels::ChannelView,
    ) -> Vec<(String, Rect)> {
        draw_row_with(width, lufs, fps, view, None)
    }

    fn draw_row_with(
        width: f32,
        lufs: Option<&str>,
        fps: &str,
        view: channels::ChannelView,
        analysis: Option<&str>,
    ) -> Vec<(String, Rect)> {
        let r = plot(width);
        let lufs = lufs.map(|s| s.to_string());
        let fps = fps.to_string();
        let analysis = analysis.map(|s| s.to_string());
        let shapes = capture(move |p| {
            let row = TopRow::plan(p, r, lufs.as_deref(), &fps, analysis.as_deref());
            let w = SpectrumWindow::new();
            // The legend, through the production function.
            match view {
                channels::ChannelView::Left => {
                    w.label_channel(p, Pos2::new(row.legend_x, row.y), row.limit_x, "L", CHANNEL_L_COLOR);
                }
                channels::ChannelView::Overlay => {
                    let used = w.label_channel(
                        p,
                        Pos2::new(row.legend_x, row.y),
                        row.limit_x,
                        "L",
                        CHANNEL_L_COLOR,
                    );
                    w.label_channel(
                        p,
                        Pos2::new(row.legend_x + used + 6.0, row.y),
                        row.limit_x,
                        "R",
                        CHANNEL_R_COLOR,
                    );
                }
                _ => {}
            }
            // Then the two fixed readouts, as the caller draws them.
            p.text(
                Pos2::new(row.fps_x, row.y),
                egui::Align2::RIGHT_TOP,
                &fps,
                readout_font(),
                Color32::WHITE,
            );
            if let Some(l) = &lufs {
                p.text(
                    Pos2::new(row.lufs_x, row.y),
                    egui::Align2::LEFT_TOP,
                    l,
                    readout_font(),
                    Color32::WHITE,
                );
            }
            // The badge, positioned and sized exactly as production does it.
            if let Some(t) = &analysis {
                let pos = Pos2::new(row.analysis_x, row.y + 2.0);
                let gal = p.layout_no_wrap(t.clone(), badge_font(), Color32::WHITE);
                let bg = Rect::from_min_size(
                    Pos2::new(pos.x - gal.size().x - BADGE_PAD, pos.y - 3.0),
                    gal.size() + egui::vec2(2.0 * BADGE_PAD, 6.0),
                );
                // Drawn as a filled rectangle in production; represented here by
                // a text shape carrying the same box, so one disjointness check
                // covers everything in the row.
                p.rect_filled(bg, 3.0, Color32::from_rgba_unmultiplied(0, 0, 0, 120));
                p.galley(
                    Pos2::new(bg.left() + BADGE_PAD, bg.top() + 3.0),
                    gal,
                    Color32::WHITE,
                );
            }
        });
        let mut out = boxes(&shapes);
        // The badge is one visual object -- a background with its text inside
        // it -- so it becomes one item, sized by the background, which is the
        // part that actually collides. Comparing its own text against its own
        // background would report an overlap that is the design.
        let badge_bg: Option<Rect> = shapes.iter().find_map(|c| match &c.shape {
            egui::Shape::Rect(r) if r.rect.width() > 20.0 => Some(r.rect),
            _ => None,
        });
        if let Some(bg) = badge_bg {
            out.retain(|(_, b)| !bg.contains_rect(*b));
            out.push(("<badge>".to_string(), bg));
        }
        out
    }

    fn assert_disjoint(items: &[(String, Rect)], what: &str) {
        for i in 0..items.len() {
            for j in (i + 1)..items.len() {
                let (an, ar) = &items[i];
                let (bn, br) = &items[j];
                assert!(
                    !ar.intersects(*br),
                    "{what}: {an:?} at {ar:?} overlaps {bn:?} at {br:?}"
                );
            }
        }
    }

    #[test]
    fn the_readouts_do_not_overlap_each_other() {
        // The default window is 700 px wide; the widest LUFS string and a
        // three-digit frame rate are the worst case.
        for width in [340.0f32, 500.0, 700.0, 900.0, 1400.0] {
            for lufs in [Some("-23.4 LUFS"), Some("— LUFS"), None] {
                for fps in ["9 fps", "60 fps", "240 fps"] {
                    for view in [
                        channels::ChannelView::Left,
                        channels::ChannelView::Overlay,
                        channels::ChannelView::Mix,
                    ] {
                        let items = draw_row(width, lufs, fps, view);
                        assert_disjoint(
                            &items,
                            &format!("width {width}, lufs {lufs:?}, fps {fps}, {view:?}"),
                        );
                    }
                }
            }
        }
    }

    /// The specific collision the owner reported: the LUFS number and the
    /// channel legend, at the size the window actually opens at.
    #[test]
    fn lufs_and_the_channel_legend_are_side_by_side() {
        let items = draw_row(700.0, Some("-23.4 LUFS"), "180 fps", channels::ChannelView::Left);
        let lufs = items
            .iter()
            .find(|(t, _)| t.contains("LUFS"))
            .expect("no LUFS readout");
        let legend = items.iter().find(|(t, _)| t == "L").expect("no legend");
        let fps = items
            .iter()
            .find(|(t, _)| t.contains("fps"))
            .expect("no fps readout");
        assert!(
            lufs.1.right() <= legend.1.left(),
            "LUFS {:?} still runs into the legend {:?}",
            lufs.1,
            legend.1
        );
        assert!(
            legend.1.right() <= fps.1.left(),
            "the legend {:?} still runs into the frame rate {:?}",
            legend.1,
            fps.1
        );
        // All three on the same line, which is the point of the row.
        assert!((lufs.1.top() - legend.1.top()).abs() < 4.0);
        assert!((lufs.1.top() - fps.1.top()).abs() < 4.0);
    }


    /// The owner's second report: the analysis badge landed on the frame rate.
    ///
    /// It was anchored to the widget rectangle's top-right corner, which after
    /// the row was introduced is where the frame rate sits. The badge is the
    /// widest thing in the row and it comes and goes, so it takes the right end
    /// and the frame rate moves left of it — rather than the frame rate jumping
    /// sideways whenever an analysis starts.
    #[test]
    fn the_analysis_badge_does_not_land_on_the_frame_rate() {
        for width in [420.0f32, 700.0, 900.0, 1400.0] {
            for analysis in [
                "analysing 0%",
                "analysing 47%  ~1m 12s",
                "analysing 100%  ~12m 30s",
            ] {
                for lufs in [Some("-23.4 LUFS"), None] {
                    for view in [channels::ChannelView::Left, channels::ChannelView::Mix] {
                        let items = draw_row_with(width, lufs, "240 fps", view, Some(analysis));
                        assert_disjoint(
                            &items,
                            &format!("width {width}, {analysis:?}, lufs {lufs:?}, {view:?}"),
                        );
                        // And the badge really was drawn, so this is not
                        // passing by drawing nothing.
                        assert!(
                            items.iter().any(|(t, _)| t == "<badge>"),
                            "no badge at width {width}"
                        );
                        assert!(
                            items.iter().any(|(t, _)| t.contains("fps")),
                            "no frame rate at width {width}"
                        );
                    }
                }
            }
        }
    }

    /// The frame rate stays put when no analysis is running, and only moves
    /// left when one starts. A readout that jumped about would be worse than
    /// the overlap.
    #[test]
    fn the_frame_rate_only_moves_to_make_room_for_the_badge() {
        let idle = draw_row_with(900.0, Some("-23.4 LUFS"), "60 fps", channels::ChannelView::Left, None);
        let busy = draw_row_with(
            900.0,
            Some("-23.4 LUFS"),
            "60 fps",
            channels::ChannelView::Left,
            Some("analysing 47%  ~1m 12s"),
        );
        let fps_of = |v: &Vec<(String, Rect)>| {
            v.iter().find(|(t, _)| t.contains("fps")).expect("no fps").1
        };
        let a = fps_of(&idle);
        let b = fps_of(&busy);
        assert!(
            b.right() < a.right(),
            "the frame rate should shift left for the badge: {a:?} then {b:?}"
        );
        // The LUFS readout does not move; it is anchored to the other end.
        let lufs_of = |v: &Vec<(String, Rect)>| {
            v.iter().find(|(t, _)| t.contains("LUFS")).expect("no LUFS").1
        };
        assert_eq!(lufs_of(&idle).left(), lufs_of(&busy).left());
    }
    /// The row lives inside the plot, so the dB margin stays free for the axis
    /// labels a Diff plot draws there.
    #[test]
    fn the_row_does_not_intrude_on_the_db_margin() {
        let r = plot(900.0);
        let items = draw_row(900.0, Some("-23.4 LUFS"), "180 fps", channels::ChannelView::Left);
        for (t, b) in &items {
            assert!(
                b.left() >= r.left(),
                "{t:?} at {b:?} reaches into the dB margin left of {}",
                r.left()
            );
            assert!(b.right() <= r.right() + 0.5, "{t:?} at {b:?} runs off the plot");
        }
    }

    /// When there is no room the legend is dropped rather than drawn over the
    /// frame rate. A missing legend is recoverable by widening the window; two
    /// numbers on top of each other are not readable at all.
    #[test]
    fn a_window_too_narrow_for_the_legend_drops_it() {
        let items = draw_row(150.0, Some("-23.4 LUFS"), "240 fps", channels::ChannelView::Left);
        assert_disjoint(&items, "narrow");
        assert!(
            !items.iter().any(|(t, _)| t == "L"),
            "the legend was drawn with no room for it: {items:?}"
        );
        // The two numbers still are.
        assert!(items.iter().any(|(t, _)| t.contains("LUFS")));
        assert!(items.iter().any(|(t, _)| t.contains("fps")));
    }

    /// Overlay draws two legends, and the second follows the measured width of
    /// the first rather than a fixed 14 px guess.
    #[test]
    fn the_overlay_legend_pair_never_touches() {
        let items = draw_row(900.0, Some("-23.4 LUFS"), "60 fps", channels::ChannelView::Overlay);
        assert_disjoint(&items, "overlay");
        let l = items.iter().find(|(t, _)| t == "L").expect("no L");
        let r = items.iter().find(|(t, _)| t == "R").expect("no R");
        assert!(l.1.right() <= r.1.left(), "L {:?} runs into R {:?}", l.1, r.1);
    }
}

// ---------------------------------------------------------------------------
// The Diff view, as it actually paints
// ---------------------------------------------------------------------------

/// These drive the production renderers through a real `egui::Painter` and read
/// back the shapes they emitted.
///
/// Asserting on `DiffLayout::sign()` and `end_labels()` is what let the first
/// cut ship: every one of those assertions passed while the plot drew ordinary
/// −80…0 dB gridlines across a ±20 dB axis, put the frequency labels along the
/// edge that was showing decibels, and repainted a band cyan when the sides
/// were swapped. The layout arithmetic was never the broken part. So these
/// tests look at emitted geometry and colour instead.
#[cfg(test)]
mod diff_paint_tests {
    use super::*;

    /// The rectangle the spectrum widget is given, and the clip the production
    /// painter uses: `ui.painter_at(rect)`.
    fn outer() -> Rect {
        Rect::from_min_size(Pos2::new(12.0, 8.0), egui::vec2(900.0, 460.0))
    }

    /// The plot inside it, derived exactly as production derives it. Note that
    /// it reaches `outer`'s top and right sides — there is no headroom above a
    /// difference axis's positive end, nor to the right of a vertical one.
    fn rect() -> Rect {
        let r = outer();
        Rect::from_min_max(
            Pos2::new(r.left() + DB_MARGIN, r.top()),
            Pos2::new(r.right(), r.bottom() - FREQ_MARGIN),
        )
    }

    /// Run `f` against a real painter clipped the way production clips, and
    /// return everything it drew **with the clip it was drawn under**.
    ///
    /// Keeping `clip_rect` is the point: a label outside it is invisible, and
    /// discarding it is what let the endpoint labels pass while being cut off.
    fn capture(f: impl FnOnce(&egui::Painter)) -> Vec<egui::epaint::ClippedShape> {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(
                Pos2::ZERO,
                egui::vec2(1200.0, 800.0),
            )),
            ..Default::default()
        };
        // `run` takes an FnMut and may call the body more than once, so the
        // FnOnce is parked in an Option and taken on the first pass.
        let mut f = Some(f);
        let out = ctx.run(input, |ctx| {
            let painter = ctx
                .layer_painter(egui::LayerId::new(
                    egui::Order::Foreground,
                    egui::Id::new("diff_paint_tests"),
                ))
                .with_clip_rect(outer());
            if let Some(f) = f.take() {
                f(&painter);
            }
        });
        out.shapes
    }

    /// Every label, with its box and the clip it was drawn under.
    fn labels(shapes: &[egui::epaint::ClippedShape]) -> Vec<(String, Rect, Color32, Rect)> {
        shapes
            .iter()
            .filter_map(|c| match &c.shape {
                egui::Shape::Text(t) => {
                    let color = t.override_text_color.unwrap_or_else(|| {
                        t.galley
                            .job
                            .sections
                            .first()
                            .map(|sec| sec.format.color)
                            .unwrap_or(Color32::PLACEHOLDER)
                    });
                    Some((
                        t.galley.text().to_string(),
                        Rect::from_min_size(t.pos, t.galley.size()),
                        color,
                        c.clip_rect,
                    ))
                }
                _ => None,
            })
            .collect()
    }

    /// Every text the shapes carry, as (text, its box, colour).
    ///
    /// The box rather than the raw `pos`, because `TextShape::pos` is the
    /// top-left *after* alignment — it already has the galley's own width and
    /// height folded in, so two differently-aligned labels are not comparable
    /// by position alone. Callers compare centres, or edges against the plot.
    fn texts(shapes: &[egui::epaint::ClippedShape]) -> Vec<(String, Rect, Color32)> {
        labels(shapes).into_iter().map(|(a, b, c, _)| (a, b, c)).collect()
    }

    fn find<'a>(
        t: &'a [(String, Rect, Color32)],
        want: &str,
    ) -> &'a (String, Rect, Color32) {
        t.iter()
            .find(|(s, _, _)| s == want)
            .unwrap_or_else(|| panic!("no text {want:?} among {:?}", t.iter().map(|x| &x.0).collect::<Vec<_>>()))
    }

    fn segments(shapes: &[egui::epaint::ClippedShape]) -> Vec<([Pos2; 2], Stroke)> {
        shapes
            .iter()
            .filter_map(|c| match &c.shape {
                egui::Shape::LineSegment { points, stroke } => Some((*points, *stroke)),
                _ => None,
            })
            .collect()
    }

    /// Every mesh vertex, as (position, colour).
    fn verts(shapes: &[egui::epaint::ClippedShape]) -> Vec<(Pos2, Color32)> {
        let mut out = Vec::new();
        for c in shapes {
            if let egui::Shape::Mesh(m) = &c.shape {
                out.extend(m.vertices.iter().map(|v| (v.pos, v.color)));
            }
        }
        out
    }

    fn axes(layout: channels::DiffLayout) -> Vec<egui::epaint::ClippedShape> {
        axes_with(layout, 48_000, 20.0, 24_000.0, freq_scale::FreqScale::Log)
    }

    fn axes_with(
        layout: channels::DiffLayout,
        sr: u32,
        min_freq: f32,
        max_freq: f32,
        scale: freq_scale::FreqScale,
    ) -> Vec<egui::epaint::ClippedShape> {
        capture(move |p| draw_diff_axes(p, rect(), layout, sr, min_freq, max_freq, scale))
    }

    fn horizontal() -> channels::DiffLayout {
        channels::DiffLayout {
            orientation: channels::DiffOrientation::Horizontal,
            ..Default::default()
        }
    }

    fn vertical() -> channels::DiffLayout {
        channels::DiffLayout {
            orientation: channels::DiffOrientation::Vertical,
            ..Default::default()
        }
    }

    // ── orientation ────────────────────────────────────────────────────────

    #[test]
    fn horizontal_diff_runs_frequency_across_and_difference_up() {
        let r = rect();
        let shapes = axes(horizontal());
        let segs = segments(&shapes);

        // The zero line spans the full width at the vertical centre.
        let zero = segs
            .iter()
            .find(|(pts, st)| {
                (pts[0].y - r.center().y).abs() < 0.5
                    && (pts[1].y - r.center().y).abs() < 0.5
                    && st.width >= 1.0
            })
            .expect("no full-width zero line at the vertical centre");
        assert!((zero.0[0].x - r.left()).abs() < 0.5, "zero line starts at {:?}", zero.0[0]);
        assert!((zero.0[1].x - r.right()).abs() < 0.5, "zero line ends at {:?}", zero.0[1]);

        let t = texts(&shapes);
        // Difference labels sit in the left margin and differ in y.
        let plus = find(&t, "R +20");
        let minus = find(&t, "L +20");
        let zero_lbl = find(&t, "0");
        for lbl in [plus, minus, zero_lbl] {
            assert!(
                lbl.1.right() <= r.left(),
                "{:?} is not in the left margin",
                lbl
            );
        }
        assert!(
            plus.1.center().y < zero_lbl.1.center().y,
            "the positive end must be above zero"
        );
        assert!(
            minus.1.center().y > zero_lbl.1.center().y,
            "the negative end must be below zero"
        );

        // Frequency labels sit under the plot and differ in x.
        let f100 = find(&t, "100");
        let f10k = find(&t, "10k");
        for lbl in [f100, f10k] {
            assert!(lbl.1.top() >= r.bottom(), "{:?} is not under the plot", lbl);
        }
        assert!(
            f100.1.center().x < f10k.1.center().x,
            "frequency must increase to the right"
        );
    }

    #[test]
    fn vertical_diff_swaps_both_axes() {
        let r = rect();
        let shapes = axes(vertical());
        let segs = segments(&shapes);

        let zero = segs
            .iter()
            .find(|(pts, st)| {
                (pts[0].x - r.center().x).abs() < 0.5
                    && (pts[1].x - r.center().x).abs() < 0.5
                    && st.width >= 1.0
            })
            .expect("no full-height zero line at the horizontal centre");
        assert!((zero.0[0].y - r.top()).abs() < 0.5);
        assert!((zero.0[1].y - r.bottom()).abs() < 0.5);

        let t = texts(&shapes);
        // Now the difference labels are under the plot, differing in x...
        let plus = find(&t, "R +20");
        let minus = find(&t, "L +20");
        for lbl in [plus, minus] {
            assert!(lbl.1.top() >= r.bottom(), "{:?} is not under the plot", lbl);
        }
        assert!(
            plus.1.center().x > minus.1.center().x,
            "positive must be to the right"
        );

        // ...and the frequency labels are in the left margin, differing in y.
        let f100 = find(&t, "100");
        let f10k = find(&t, "10k");
        for lbl in [f100, f10k] {
            assert!(
                lbl.1.right() <= r.left(),
                "{:?} is not in the left margin",
                lbl
            );
        }
        assert!(
            (f100.1.center().y - f10k.1.center().y).abs() > 20.0,
            "frequency labels must be spread down the side, got {:?} and {:?}",
            f100,
            f10k
        );
    }

    // ── frequency reversal moves the labels too ────────────────────────────

    #[test]
    fn flipping_frequency_moves_the_labels_and_the_bars_the_same_way() {
        let r = rect();
        let plain = horizontal();
        let flipped = channels::DiffLayout { flip_frequency: true, ..plain };

        let a = texts(&axes(plain));
        let b = texts(&axes(flipped));
        let x0 = find(&a, "1k").1.center().x;
        let x1 = find(&b, "1k").1.center().x;
        let mirror = r.left() + r.right() - x0;
        assert!(
            (x1 - mirror).abs() < 1.0,
            "the 1k tick should mirror about the plot centre: {x0} -> {x1}, expected {mirror}"
        );

        // And the bars move with it. One loud band near the bottom of the
        // spectrum; find where its colour lands in each layout.
        let n = 64;
        let mut left = vec![0.5f32; n];
        let right = vec![0.5f32; n];
        left[2] = 0.0; // right much louder in band 2
        let bar_x = |layout| {
            let sh = capture(|p| draw_channel_diff(p, &left, &right, r, 0.0, layout));
            let v = verts(&sh);
            let xs: Vec<f32> = v
                .iter()
                .filter(|(_, c)| *c == CHANNEL_R_COLOR)
                .map(|(p, _)| p.x)
                .collect();
            assert!(!xs.is_empty(), "no right-coloured bar was drawn");
            xs.iter().sum::<f32>() / xs.len() as f32
        };
        let bx0 = bar_x(plain);
        let bx1 = bar_x(flipped);
        let bar_mirror = r.left() + r.right() - bx0;
        assert!(
            (bx1 - bar_mirror).abs() < 6.0,
            "the bar should mirror with the labels: {bx0} -> {bx1}, expected {bar_mirror}"
        );
    }

    // ── colour identity ────────────────────────────────────────────────────

    #[test]
    fn swapping_the_sides_moves_a_band_without_recolouring_it() {
        let r = rect();
        let n = 32;
        // Left louder everywhere: every bar belongs to left, in cyan.
        let left = vec![0.9f32; n];
        let right = vec![0.2f32; n];

        for (layout, expect_above) in [
            (horizontal(), false),
            (
                channels::DiffLayout { flip_channels: true, ..horizontal() },
                true,
            ),
        ] {
            let sh = capture(|p| draw_channel_diff(p, &left, &right, r, 0.0, layout));
            let v = verts(&sh);
            assert!(!v.is_empty(), "nothing drawn");
            for (_, c) in &v {
                assert_eq!(
                    *c, CHANNEL_L_COLOR,
                    "left-louder audio must stay cyan whichever side it is drawn on"
                );
            }
            let mid = r.center().y;
            let above = v.iter().filter(|(p, _)| p.y < mid - 0.5).count();
            let below = v.iter().filter(|(p, _)| p.y > mid + 0.5).count();
            if expect_above {
                assert!(above > 0 && below == 0, "swap should put left above the line");
            } else {
                assert!(below > 0 && above == 0, "left belongs below the line by default");
            }
        }
    }

    #[test]
    fn the_end_labels_carry_their_own_channel_colour() {
        for (layout, pos_text, pos_color, neg_text, neg_color) in [
            (horizontal(), "R +20", CHANNEL_R_COLOR, "L +20", CHANNEL_L_COLOR),
            (
                channels::DiffLayout { flip_channels: true, ..horizontal() },
                "L +20",
                CHANNEL_L_COLOR,
                "R +20",
                CHANNEL_R_COLOR,
            ),
        ] {
            let t = texts(&axes(layout));
            let pos = find(&t, pos_text);
            let neg = find(&t, neg_text);
            assert_eq!(pos.2, pos_color, "{pos_text} must keep its channel colour");
            assert_eq!(neg.2, neg_color, "{neg_text} must keep its channel colour");
            // And the one named at the positive end really is at the top.
            assert!(
                pos.1.center().y < neg.1.center().y,
                "{pos_text} should be above {neg_text}"
            );
        }
    }

    // ── the zero label and the scale ───────────────────────────────────────

    #[test]
    fn the_difference_axis_is_labelled_in_signed_decibels_about_zero() {
        let t = texts(&axes(horizontal()));
        let names: Vec<&str> = t.iter().map(|(s, _, _)| s.as_str()).collect();
        for want in ["0", "+10", "-10", "R +20", "L +20"] {
            assert!(names.contains(&want), "missing {want} among {names:?}");
        }
        // The ordinary plot's level ticks must not be here: this axis has no
        // −70 dB on it.
        for unwanted in ["-70", "-80", "-40"] {
            assert!(
                !names.contains(&unwanted),
                "{unwanted} is a level-axis tick and does not belong on a difference axis"
            );
        }
    }

    #[test]
    fn a_warped_frequency_scale_moves_the_ticks_with_the_bars() {
        // Under `Log` the tick is where a bare logarithm would put it; under a
        // warped scale it must move, or the label names the wrong bar.
        let r = rect();
        let at = |scale| {
            let sh = capture(|p| {
                draw_diff_axes(p, r, horizontal(), 48_000, 20.0, 24_000.0, scale)
            });
            find(&texts(&sh), "1k").1.center().x
        };
        let log = at(freq_scale::FreqScale::Log);
        let erb = at(freq_scale::FreqScale::Erb);
        assert!(
            (log - erb).abs() > 1.0,
            "the ERB scale must move the 1k tick away from its log position \
             (log {log}, erb {erb})"
        );
        // And it lands where the scale says, not somewhere invented.
        let want = r.left()
            + freq_scale::FreqScale::Erb.position_of(1000.0, 20.0, 24_000.0) * r.width();
        assert!((erb - want).abs() < 1.0, "erb tick at {erb}, scale says {want}");
    }


    // ── the label must be inside the clip to exist at all ──────────────────

    /// Every label the axes draw, in every arrangement, wholly inside the clip.
    ///
    /// The production painter is `ui.painter_at(rect)` and the plot reaches that
    /// rectangle's top and right sides, so a centred label anchored on the
    /// positive end of a difference axis had half of itself cut off. The
    /// endpoints are the two that matter and they are always on an edge — that
    /// is what makes them endpoints — but this sweeps all of them.
    #[test]
    fn every_axis_label_is_wholly_inside_the_clip() {
        for orientation in [
            channels::DiffOrientation::Horizontal,
            channels::DiffOrientation::Vertical,
        ] {
            for flip_channels in [false, true] {
                for flip_frequency in [false, true] {
                    let layout = channels::DiffLayout {
                        orientation,
                        flip_channels,
                        flip_frequency,
                    };
                    for (sr, lo, hi) in
                        [(48_000u32, 20.0f32, 24_000.0f32), (44_100, 20.0, 24_000.0), (96_000, 500.0, 20_000.0)]
                    {
                        let shapes =
                            axes_with(layout, sr, lo, hi, freq_scale::FreqScale::Log);
                        let ls = labels(&shapes);
                        assert!(!ls.is_empty(), "{layout:?} {sr}: nothing drawn");
                        for (text, bounds, _, clip) in &ls {
                            assert!(
                                clip.contains_rect(*bounds),
                                "{layout:?} sr={sr} {lo}-{hi}: label {text:?} at {bounds:?} \
                                 is not inside the clip {clip:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The two Codex named, checked by name rather than by sweep, so a
    /// regression says which one came back.
    #[test]
    fn the_endpoint_labels_are_not_cut_off() {
        // Horizontal: the positive end sits on the plot's top edge, which is
        // also the clip's top edge.
        for (orientation, who) in [
            (channels::DiffOrientation::Horizontal, "upper"),
            (channels::DiffOrientation::Vertical, "right"),
        ] {
            for flip_channels in [false, true] {
                let layout = channels::DiffLayout {
                    orientation,
                    flip_channels,
                    ..Default::default()
                };
                let ls = labels(&axes(layout));
                let (pos_ch, neg_ch) = layout.end_channels();
                for ch in [pos_ch, neg_ch] {
                    let want = format!("{} +20", ch.label());
                    let (_, bounds, _, clip) = ls
                        .iter()
                        .find(|(t, _, _, _)| *t == want)
                        .unwrap_or_else(|| panic!("{who} end: no label {want:?}"));
                    assert!(
                        clip.contains_rect(*bounds),
                        "{who} end, swap={flip_channels}: {want:?} at {bounds:?} \
                         escapes the clip {clip:?}"
                    );
                }
            }
        }
    }

    // ── the frequency axis must agree with the bars ────────────────────────

    /// Positions of the frequency tick marks, along the frequency axis.
    ///
    /// The tick mark, not the label. A label near the end of an axis is nudged
    /// inward so it stays inside the clip, which moves its box but not the
    /// position it names; the 4 px tick is the placement itself. Measuring the
    /// label would make a correct nudge look like a misplaced axis.
    fn tick_positions(
        shapes: &[egui::epaint::ClippedShape],
        vertical: bool,
        r: Rect,
    ) -> Vec<f32> {
        let mut out: Vec<f32> = segments(shapes)
            .into_iter()
            .filter_map(|(pts, _)| {
                if vertical {
                    // A short horizontal stub in the left margin.
                    let on = (pts[0].x - (r.left() - 4.0)).abs() < 0.5
                        && (pts[1].x - r.left()).abs() < 0.5
                        && (pts[0].y - pts[1].y).abs() < 0.5;
                    on.then_some(pts[0].y)
                } else {
                    // A short vertical stub under the plot.
                    let on = (pts[0].y - r.bottom()).abs() < 0.5
                        && (pts[1].y - (r.bottom() + 4.0)).abs() < 0.5
                        && (pts[0].x - pts[1].x).abs() < 0.5;
                    on.then_some(pts[0].x)
                }
            })
            .collect();
        out.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out
    }

    /// Where the renderer actually puts the bar holding `freq`, as a fraction of
    /// the frequency axis.
    ///
    /// Derived from `bar_center` — the function the analyser and the loudness
    /// weighting use to ask what frequency a bar sits at — and deliberately not
    /// from `position_of`, which is what the label code itself calls. Checking a
    /// formula against itself would pass on the broken code too.
    fn bar_fraction_of(
        freq: f32,
        min_freq: f32,
        max_freq: f32,
        scale: freq_scale::FreqScale,
        n: usize,
    ) -> f32 {
        let mut best = 0usize;
        let mut best_d = f32::MAX;
        for i in 0..n {
            let c = scale.bar_center(i, n, min_freq, max_freq);
            let d = (c.log10() - freq.log10()).abs();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        (best as f32 + 0.5) / n as f32
    }

    /// A tick must land on the bar it names, at any sample rate.
    ///
    /// 44.1 kHz with the default 24 kHz ceiling is the case that was wrong:
    /// positions were computed over `min_freq..nyquist` while the bars are laid
    /// out over `min_freq..max_freq`, so every label sat to the right of its
    /// bar. 48 kHz hid it, because there Nyquist *is* the ceiling.
    #[test]
    fn ticks_land_on_the_bars_they_name() {
        const N: usize = 512;
        let r = rect();
        for scale in [
            freq_scale::FreqScale::Log,
            freq_scale::FreqScale::Erb,
            freq_scale::FreqScale::Blend(0.5),
        ] {
            for (sr, lo, hi) in [
                (44_100u32, 20.0f32, 24_000.0f32), // max_freq above Nyquist
                (48_000, 20.0, 24_000.0),          // Nyquist == ceiling
                (96_000, 20.0, 24_000.0),          // Nyquist above ceiling
                (44_100, 500.0, 20_000.0),         // raised floor
            ] {
                let shapes = axes_with(
                    channels::DiffLayout::default(),
                    sr,
                    lo,
                    hi,
                    scale,
                );
                let ls = texts(&shapes);
                let marks = tick_positions(&shapes, false, r);
                for &freq in FREQ_TICKS {
                    let nyq = sr as f32 / 2.0;
                    if freq < lo || freq > nyq.min(hi) {
                        continue;
                    }
                    let name = tick_label(freq);
                    assert!(
                        ls.iter().any(|(t, _, _)| *t == name),
                        "{scale:?} sr={sr} {lo}-{hi}: no label {name:?}"
                    );
                    let want = r.left() + bar_fraction_of(freq, lo, hi, scale, N) * r.width();
                    // Within one bar of the 512-bar grid.
                    let tol = r.width() / N as f32 + 1.0;
                    assert!(
                        marks.iter().any(|&m| (m - want).abs() <= tol),
                        "{scale:?} sr={sr} {lo}-{hi}: no tick mark for {name:?} near \
                         {want} (tolerance {tol}); marks are {marks:?}"
                    );
                }
            }
        }
    }

    /// The same thing measured against drawn geometry rather than arithmetic:
    /// put a spike in one bar, and check the tick naming that frequency lands
    /// inside the bar that was actually painted.
    #[test]
    fn a_tick_sits_inside_the_bar_it_names() {
        const N: usize = 64;
        let r = rect();
        let (sr, lo, hi) = (44_100u32, 20.0f32, 24_000.0f32);
        let scale = freq_scale::FreqScale::Log;
        let freq = 1000.0f32;

        // The bar whose centre is nearest 1 kHz, by `bar_center`.
        let mut bar = 0usize;
        let mut bd = f32::MAX;
        for i in 0..N {
            let d = (scale.bar_center(i, N, lo, hi).log10() - freq.log10()).abs();
            if d < bd {
                bd = d;
                bar = i;
            }
        }

        // Right much louder in exactly that bar.
        let mut left = vec![0.5f32; N];
        let right = vec![0.5f32; N];
        left[bar] = 0.0;
        let layout = channels::DiffLayout::default();
        let painted = capture(move |p| draw_channel_diff(p, &left, &right, r, 0.0, layout));
        let xs: Vec<f32> = verts(&painted)
            .into_iter()
            .filter(|(_, c)| *c == CHANNEL_R_COLOR)
            .map(|(p, _)| p.x)
            .collect();
        assert!(!xs.is_empty(), "no bar was painted for the spike");
        let (bar_lo, bar_hi) = (
            xs.iter().cloned().fold(f32::MAX, f32::min),
            xs.iter().cloned().fold(f32::MIN, f32::max),
        );

        let shapes = axes_with(layout, sr, lo, hi, scale);
        assert!(
            texts(&shapes).iter().any(|(t, _, _)| t == "1k"),
            "no 1k label"
        );
        let marks = tick_positions(&shapes, false, r);
        assert!(
            marks.iter().any(|&x| x >= bar_lo - 1.0 && x <= bar_hi + 1.0),
            "no 1 kHz tick mark inside the bar painted over {bar_lo}..{bar_hi}; \
             marks are {marks:?}"
        );
    }

    /// A raised lower bound removes the ticks below it. It must not pile them
    /// onto the left endpoint, which reads as a real tick at a frequency the
    /// axis does not cover.
    #[test]
    fn a_raised_minimum_drops_the_lower_ticks_rather_than_stacking_them() {
        for orientation in [
            channels::DiffOrientation::Horizontal,
            channels::DiffOrientation::Vertical,
        ] {
            for flip_frequency in [false, true] {
                let layout = channels::DiffLayout {
                    orientation,
                    flip_frequency,
                    ..Default::default()
                };
                let ls = texts(&axes_with(
                    layout,
                    48_000,
                    500.0,
                    20_000.0,
                    freq_scale::FreqScale::Log,
                ));
                let names: Vec<&str> = ls.iter().map(|(t, _, _)| t.as_str()).collect();
                for gone in ["50", "100", "200"] {
                    assert!(
                        !names.contains(&gone),
                        "{layout:?}: {gone} Hz is below the 500 Hz minimum but was drawn"
                    );
                }
                for kept in ["500", "1k", "2k", "5k", "10k", "20k"] {
                    assert!(names.contains(&kept), "{layout:?}: {kept} is missing");
                }
                // And nothing is stacked: every tick mark has its own position.
                let vertical = orientation == channels::DiffOrientation::Vertical;
                let marks = tick_positions(
                    &axes_with(layout, 48_000, 500.0, 20_000.0, freq_scale::FreqScale::Log),
                    vertical,
                    rect(),
                );
                assert_eq!(
                    marks.len(),
                    6,
                    "{layout:?}: expected six ticks from 500 Hz up, got {marks:?}"
                );
                for w in marks.windows(2) {
                    assert!(
                        w[1] - w[0] > 4.0,
                        "{layout:?}: two ticks landed on top of each other at {w:?}"
                    );
                }
            }
        }
    }

    /// The ordinary Mix plot uses the same mapping. It has the same bug and the
    /// same fix, and it is the path most people look at.
    #[test]
    fn the_ordinary_frequency_labels_land_on_their_bars_too() {
        const N: usize = 512;
        let r = rect();
        for scale in [freq_scale::FreqScale::Log, freq_scale::FreqScale::Erb] {
            for (sr, lo, hi) in [(44_100u32, 20.0f32, 24_000.0f32), (48_000, 500.0, 20_000.0)] {
                let shapes = capture(move |p| draw_freq_labels(p, r, sr, lo, hi, scale));
                let ls = labels(&shapes);
                for (text, bounds, _, clip) in &ls {
                    assert!(
                        clip.contains_rect(*bounds),
                        "ordinary axis: {text:?} escapes the clip"
                    );
                }
                let names: Vec<&str> = ls.iter().map(|(t, _, _, _)| t.as_str()).collect();
                if lo > 100.0 {
                    assert!(!names.contains(&"50"), "50 Hz drawn below a 500 Hz minimum");
                }
                let marks = tick_positions(&shapes, false, r);
                for &freq in FREQ_TICKS {
                    let nyq = sr as f32 / 2.0;
                    if freq < lo || freq > nyq.min(hi) {
                        continue;
                    }
                    let name = tick_label(freq);
                    assert!(
                        ls.iter().any(|(t, _, _, _)| *t == name),
                        "ordinary axis {scale:?} sr={sr}: no label {name:?}"
                    );
                    let want = r.left() + bar_fraction_of(freq, lo, hi, scale, N) * r.width();
                    let tol = r.width() / N as f32 + 1.0;
                    assert!(
                        marks.iter().any(|&m| (m - want).abs() <= tol),
                        "ordinary axis {scale:?} sr={sr}: no tick mark for {name:?} near \
                         {want}; marks are {marks:?}"
                    );
                }
            }
        }
    }

    /// Nyquist decides which ticks exist; it does not decide where they go.
    ///
    /// Stated as its own test because the two bounds were conflated, and the
    /// distinction is invisible whenever they happen to be equal.
    #[test]
    fn nyquist_filters_ticks_without_rescaling_the_axis() {
        let scale = freq_scale::FreqScale::Log;
        // Same display range, three sample rates. The ticks that survive differ;
        // the positions of the ones that survive do not.
        let a = eligible_ticks(96_000, 20.0, 24_000.0, scale);
        let b = eligible_ticks(48_000, 20.0, 24_000.0, scale);
        let c = eligible_ticks(44_100, 20.0, 24_000.0, scale);
        assert_eq!(a.len(), b.len(), "48 kHz should keep every tick 96 kHz does");
        assert_eq!(a.len(), c.len(), "20 kHz is under 22.05 kHz, so all survive");
        for ((f1, t1), (f2, t2)) in a.iter().zip(&c) {
            assert_eq!(f1, f2);
            assert!(
                (t1 - t2).abs() < 1e-6,
                "{f1} Hz moved from {t1} to {t2} when the sample rate changed"
            );
        }
        // A ceiling below a tick removes it.
        let d = eligible_ticks(16_000, 20.0, 24_000.0, scale);
        assert!(
            d.iter().all(|&(f, _)| f <= 8_000.0),
            "a tick above Nyquist survived: {d:?}"
        );
        assert!(!d.is_empty(), "everything was filtered out");
        // And the survivors are still where they were.
        for (f, t) in &d {
            let (_, t0) = a.iter().find(|(g, _)| g == f).unwrap();
            assert!(
                (t - t0).abs() < 1e-6,
                "{f} Hz moved to {t} when Nyquist dropped (was {t0})"
            );
        }
    }
    // ── the fallback ───────────────────────────────────────────────────────

    #[test]
    fn a_diff_that_fell_back_to_mix_gets_ordinary_axes() {
        use channels::{ChannelAvailability, ChannelView};
        // Every reason Diff can be refused. In each, the effective view is Mix
        // and the plot must not be given difference axes.
        for avail in [
            ChannelAvailability::Mono,
            ChannelAvailability::Multichannel(6),
            ChannelAvailability::NoLiveTap,
            ChannelAvailability::PreProcessMono,
            ChannelAvailability::UnsupportedStyle,
        ] {
            let eff = channels::effective_view(ChannelView::Diff, &avail);
            assert_eq!(eff, ChannelView::Mix, "{avail:?} should fall back");
            assert!(
                !uses_diff_axes(eff, true),
                "{avail:?} fell back to Mix but still asked for difference axes"
            );
        }
        // Available and drawn: difference axes.
        let ok = ChannelAvailability::Available;
        let eff = channels::effective_view(ChannelView::Diff, &ok);
        assert_eq!(eff, ChannelView::Diff);
        assert!(uses_diff_axes(eff, true));
        // Available but the renderer bailed on a missing channel: ordinary axes.
        assert!(
            !uses_diff_axes(eff, false),
            "the renderer drew nothing, so the plot underneath is not a Diff"
        );
        // And no other view claims them.
        for view in ChannelView::ALL {
            if view == ChannelView::Diff {
                continue;
            }
            assert!(!uses_diff_axes(view, true), "{view:?} is not a Diff");
        }
    }
}

/// Right minus left, on its own axis. The arithmetic lives in `channels` and
/// has its own tests; these check the wiring — that the view is a channel view
/// like the others, refuses to draw without both channels, and is reachable.
#[cfg(test)]
mod diff_view_tests {
    use super::*;

    const BARS: usize = 32;
    const FFT: usize = 1024;

    fn tone(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / 48_000.0).sin() * 0.5)
            .collect()
    }

    fn window() -> SpectrumWindow {
        let mut w = SpectrumWindow::new();
        w.mode = SpectrumMode::RealTime;
        w.style = VizStyle::Bars;
        w.channel_view = channels::ChannelView::Diff;
        w.analyzer.sample_rate = 48_000;
        w.analyzer.fft_size = FFT;
        w.analyzer.bar_count = BARS;
        w.bar_count = BARS;
        w.analyzer.min_freq = 20.0;
        w.analyzer.max_freq = 20_000.0;
        w.analyzer.smoothing = 0.0;
        w.analyzer.rebuild_fft();
        w.analyzer.magnitudes = vec![0.0; BARS];
        w.analyzer.smoothed = vec![0.0; BARS];
        w.analyzer.peak_input = vec![0.0; BARS];
        w
    }

    fn feed(w: &SpectrumWindow, lf: f32, rf: f32) {
        begin_stereo_stream(&w.stereo_buf, 2);
        let (l, r) = (tone(lf, FFT * 2), tone(rf, FFT * 2));
        let mut g = w.stereo_buf.lock().unwrap();
        for i in 0..(FFT * 2) {
            g.frames.push([l[i], r[i]]);
        }
    }

    /// It is a channel view: it needs both, and it is offered wherever the
    /// others are.
    #[test]
    fn diff_is_offered_exactly_where_the_other_channel_views_are() {
        assert!(channels::ChannelView::ALL.contains(&channels::ChannelView::Diff));
        // Available with a live stereo tap.
        assert!(
            channels::availability(false, true, 2).is_available(),
            "a stereo tap must offer it"
        );
        // And refused everywhere the others are, for the same reasons.
        for (pre, style_ok, ch) in [(true, true, 2u16), (false, false, 2), (false, true, 1)] {
            let a = channels::availability(pre, style_ok, ch);
            assert!(!a.is_available());
            assert_eq!(
                channels::effective_view(channels::ChannelView::Diff, &a),
                channels::ChannelView::Mix,
                "Diff must fall back to Mix like every other channel view"
            );
        }
    }

    /// Hard-panned material reads as a difference, and in the right direction.
    #[test]
    fn a_panned_tone_produces_a_signed_difference() {
        let mut w = window();
        // 1 kHz on the left only.
        begin_stereo_stream(&w.stereo_buf, 2);
        {
            let l = tone(1000.0, FFT * 2);
            let mut g = w.stereo_buf.lock().unwrap();
            for &sample in &l {
                g.frames.push([sample, 0.0]);
            }
        }
        w.tick_channels(1.0 / 60.0);
        let frame = w.analyzer.channel_frame();
        let (l, r) = frame.pair().expect("a stereo tap must yield a pair");

        let worst = (0..BARS)
            .map(|b| channels::diff_fraction(l[b], r[b]))
            .fold(0.0f32, |acc, f| if f.abs() > acc.abs() { f } else { acc });
        assert!(
            worst < -0.5,
            "a left-only tone should read strongly negative, got {worst}"
        );
    }

    /// Identical channels are a flat line, which is the reading that matters
    /// most: it is what "balanced" looks like.
    #[test]
    fn identical_channels_read_as_no_difference() {
        let mut w = window();
        feed(&w, 1000.0, 1000.0);
        w.tick_channels(1.0 / 60.0);
        let frame = w.analyzer.channel_frame();
        let (l, r) = frame.pair().unwrap();
        for b in 0..BARS {
            let f = channels::diff_fraction(l[b], r[b]);
            assert!(
                f.abs() < 1e-3,
                "bar {b} of identical channels read {f}, not flat"
            );
        }
    }

    /// The layout round-trips, so a preference survives a restart.
    #[test]
    fn the_diff_layout_is_persisted() {
        let mut w = SpectrumWindow::new();
        w.diff_layout = channels::DiffLayout {
            orientation: channels::DiffOrientation::Vertical,
            flip_channels: true,
            flip_frequency: true,
        };
        let json = serde_json::to_string(&w.snapshot()).unwrap();
        let back: SpectrumSettings = serde_json::from_str(&json).unwrap();
        let mut other = SpectrumWindow::new();
        other.apply_settings(&back);
        assert_eq!(other.diff_layout, w.diff_layout);
    }

    /// A settings file written before the control existed takes the default
    /// rather than a half-initialised layout.
    #[test]
    fn a_legacy_settings_file_takes_the_default_layout() {
        let s: SpectrumSettings = serde_json::from_str(r#"{"smoothing":0.75}"#).unwrap();
        let mut w = SpectrumWindow::new();
        w.apply_settings(&s);
        assert_eq!(w.diff_layout, channels::DiffLayout::default());
        assert_eq!(
            w.diff_layout.orientation,
            channels::DiffOrientation::Horizontal
        );
    }

    /// Without both channels there is no difference to draw, and the plot falls
    /// back to the Mix rather than drawing a half-difference.
    #[test]
    fn diff_refuses_to_draw_with_one_channel() {
        let w = window();
        // Nothing fed: no channels produced.
        let frame = w.analyzer.channel_frame();
        assert!(frame.pair().is_none());
        assert!(
            channels::ChannelView::Diff.draws_left() && channels::ChannelView::Diff.draws_right(),
            "the guard in draw_channel_view depends on both being declared"
        );
    }
}

#[cfg(test)]
mod waterfall_ring_tests {
    use super::*;

    /// Every row must be shown exactly once, and at the right size.
    ///
    /// The ring is drawn as two quads and the split has to line up with where
    /// the texture wraps — get it wrong by a row and the whole display is
    /// stretched or a row is drawn twice, which is the sort of thing that looks
    /// almost right and never gets noticed.
    #[test]
    fn slices_tile_the_texture_exactly_once() {
        let h = WATERFALL_ROWS_FALLBACK;
        for head in 0..h {
            let (split, top, bot) = waterfall_slices(head, h);
            assert!((0.0..=1.0).contains(&split), "head {head}: split {split}");
            // The two v-ranges must cover [0,1] with no gap and no overlap.
            assert!((bot.0 - 0.0).abs() < 1e-6);
            assert!((top.1 - 1.0).abs() < 1e-6);
            assert!((bot.1 - top.0).abs() < 1e-6, "head {head}: gap at the seam");
            // Each quad's share of the screen must equal its share of the
            // texture, or its rows are scaled differently from the other's.
            assert!((split - (top.1 - top.0)).abs() < 1e-6,
                    "head {head}: top quad stretched");
            assert!(((1.0 - split) - (bot.1 - bot.0)).abs() < 1e-6,
                    "head {head}: bottom quad stretched");
        }
    }

    /// Rows are written backwards, so the newest sits at `head + 1` — which is
    /// exactly where the top of the screen starts reading.
    #[test]
    fn the_newest_row_lands_at_the_top() {
        let h = WATERFALL_ROWS_FALLBACK;
        // Just after writing row `t`, the head has moved back to `t - 1`.
        for t in [0usize, 1, 7, h - 1] {
            let head = (t + h - 1) % h;
            let (_, top, bot) = waterfall_slices(head, h);
            let newest_v = t as f32 / h as f32;
            // The newest row is the first row the top quad shows — unless it is
            // row 0, in which case the top quad is empty and the bottom starts there.
            let starts_at = if top.1 > top.0 { top.0 } else { bot.0 };
            assert!((starts_at - newest_v).abs() < 1e-6,
                    "row {t}: screen starts at {starts_at}, newest is at {newest_v}");
        }
    }

    /// A full wrap must return to where it started, or the ring drifts.
    #[test]
    fn writing_a_full_ring_returns_the_head() {
        let h = WATERFALL_ROWS_FALLBACK;
        let mut head = 0usize;
        for _ in 0..h { head = (head + h - 1) % h; }
        assert_eq!(head, 0);
    }
}

#[cfg(test)]
mod cache_key_tests {
    use super::*;

    fn key(bm: &BarMappingMode, cfg: &aslt::AsltConfig, fps: f32) -> String {
        cache_path_for(
            &PathBuf::from("/music/track.flac"), 1024, 8192, 16, 0.875,
            &WindowFn::Hann, 20.0, 24_000.0, bm, &InterpolationMode::None,
            crate::dsd::decimate::DEFAULT_ANALYSIS_RATE, cfg, fps,
        )
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned()
    }

    /// The failure this guards against is silent: without the superlet
    /// parameters in the key, switching Fast → Extreme reloads the Fast cache
    /// and the display never changes, which reads as "the preset does nothing"
    /// rather than as a bug.
    #[test]
    fn superlet_params_change_the_cache_key() {
        let fast = aslt::AsltPreset::Fast.config();
        let extreme = aslt::AsltPreset::Extreme.config();
        let bm = BarMappingMode::Superlet;
        assert_ne!(key(&bm, &fast, 180.0), key(&bm, &extreme, 180.0));
        // Frame rate is a free parameter here, so it has to be in the key too.
        assert_ne!(key(&bm, &fast, 180.0), key(&bm, &fast, 60.0));
    }

    /// A superlet cache must never be mistaken for an FFT one, even though the
    /// FFT knobs are zeroed out of its key.
    #[test]
    fn superlet_never_collides_with_fft_modes() {
        let cfg = aslt::AsltPreset::Standard.config();
        let slt = key(&BarMappingMode::Superlet, &cfg, 180.0);
        for bm in [BarMappingMode::FlatOverlap, BarMappingMode::Gaussian, BarMappingMode::Cqt] {
            assert_ne!(slt, key(&bm, &cfg, 180.0));
        }
    }

    /// Existing caches must stay valid: superlet settings are invisible to the
    /// FFT modes, and so is the pre-process frame rate.
    #[test]
    fn fft_keys_ignore_superlet_settings() {
        let a = aslt::AsltPreset::Fast.config();
        let b = aslt::AsltPreset::Extreme.config();
        for bm in [BarMappingMode::FlatOverlap, BarMappingMode::Gaussian, BarMappingMode::Cqt] {
            assert_eq!(key(&bm, &a, 180.0), key(&bm, &b, 60.0));
        }
    }

    /// Every stage must be named. A bar frozen at 50 % with no label was read as
    /// a hang; the same freeze labelled "Loudness/key…" is just slow.
    #[test]
    fn every_progress_value_has_a_phase() {
        assert_eq!(phase_label(0), "Decoding…");
        assert_eq!(phase_label(PROG_DECODE_END), "Decoding…");
        assert_eq!(phase_label(PROG_DECODE_END + 1), "Loudness/key…");
        assert_eq!(phase_label(PROG_LOUDNESS_END), "Loudness/key…");
        assert_eq!(phase_label(50), "Transform…");
        assert_eq!(phase_label(99), "Transform…");
        assert_eq!(phase_label(100), "Finishing…");
    }

    /// Progress must never twitch backwards, however the threads interleave.
    #[test]
    fn progress_only_moves_forward() {
        let p = Arc::new(AtomicUsize::new(0));
        for v in [10usize, 40, 25, 55, 51, 99, 70] {
            store_progress_max(&p, v);
        }
        assert_eq!(p.load(Ordering::Relaxed), 99);
    }

    /// A repeat of the track being analysed must not cancel the analysis.
    ///
    /// `on_play` fires again on every loop, and it used to call `reset()`, which
    /// aborts. A superlet run lasting longer than the track could therefore
    /// never finish: each repeat sent it back to zero, for ever.
    #[test]
    fn repeat_keeps_the_running_analysis() {
        let mut a = SpectrumAnalyzer::new(new_sample_buf());
        let (_tx, rx) = std::sync::mpsc::channel::<PreMessage>();
        a.pre_receiver = Some(rx);
        a.is_analyzing.store(true, Ordering::Relaxed);
        a.analysis_progress.store(63, Ordering::Relaxed);
        a.analyzing_path = Some(PathBuf::from("/music/track.flac"));
        let abort = Arc::clone(&a.abort_analysis);

        a.reset_keeping_analysis();

        assert!(a.pre_receiver.is_some(), "receiver dropped — the result would be stranded");
        assert!(a.is_analyzing.load(Ordering::Relaxed), "analysis flag cleared");
        assert!(!abort.load(Ordering::Relaxed), "analysis was told to abort");
        assert_eq!(a.analysis_progress.load(Ordering::Relaxed), 63, "progress was reset");
        assert!(a.analyzing_path.is_some());
    }

    /// …but a genuine track change still cancels, so two multi-minute runs never
    /// overlap.
    #[test]
    fn track_change_cancels_the_running_analysis() {
        let mut a = SpectrumAnalyzer::new(new_sample_buf());
        let (_tx, rx) = std::sync::mpsc::channel::<PreMessage>();
        a.pre_receiver = Some(rx);
        a.is_analyzing.store(true, Ordering::Relaxed);
        a.analyzing_path = Some(PathBuf::from("/music/track.flac"));
        let abort = Arc::clone(&a.abort_analysis);

        a.reset();

        assert!(abort.load(Ordering::Relaxed), "old run was not told to stop");
        assert!(a.pre_receiver.is_none());
        assert!(a.analyzing_path.is_none());
    }

    /// Measurement, not a test: try candidate cache encodings against a real
    /// file on this machine and print what each would have saved.
    ///
    /// `cargo test --release -- --ignored --nocapture cache_encoding_survey`
    #[test]
    #[ignore = "measurement — needs a populated cache, run explicitly"]
    fn cache_encoding_survey() {
        let dir = home_dir().join(".moosik").join("cache");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            println!("no cache dir at {}", dir.display());
            return;
        };
        let mut best: Option<(u64, PathBuf)> = None;
        for e in entries.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.extension().map(|x| x == "spectrumcache").unwrap_or(false)
                && let Ok(m) = e.metadata()
                && best.as_ref().is_none_or(|(len, _)| m.len() > *len)
            {
                best = Some((m.len(), p));
            }
        }
        let Some((on_disk, path)) = best else { println!("cache is empty"); return; };

        let raw = std::fs::read(&path).unwrap();
        let frames = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
        let bars = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let flat = lz4_flex::decompress_size_prepended(&raw[12..]).unwrap();
        let vals: Vec<u16> = flat
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        println!("\n{}", path.file_name().unwrap().to_string_lossy());
        println!("{frames} frames x {bars} bars = {:.1} MB raw, {:.1} MB on disk\n",
                 flat.len() as f64 / 1e6, on_disk as f64 / 1e6);

        let mb = |b: usize| b as f64 / 1e6;
        let report = |name: &str, bytes: Vec<u8>| {
            let c = lz4_flex::compress_prepend_size(&bytes).len();
            println!("{name:<34} {:>7.1} MB  ({:.2}x vs raw)", mb(c), flat.len() as f64 / c as f64);
        };

        report("current: u16 interleaved", flat.clone());

        // Delta along time: consecutive frames overlap ~97 % at 180 fps, so the
        // difference should be small even though the absolute values are not.
        let zig = |d: i32| ((d << 1) ^ (d >> 31)) as u32 as u16;
        let mut dt = Vec::with_capacity(flat.len());
        for f in 0..frames {
            for b in 0..bars {
                let cur = vals[f * bars + b] as i32;
                let prev = if f == 0 { 0 } else { vals[(f - 1) * bars + b] as i32 };
                dt.extend_from_slice(&zig(cur - prev).to_le_bytes());
            }
        }
        report("delta-in-time, zigzag", dt.clone());

        // Same, but with the two bytes of every value separated. The high byte
        // carries the picture and should compress; the low byte is mostly
        // quantisation noise and will not, so interleaving them lets the noise
        // spoil the whole stream.
        let split = |src: &[u8]| {
            let mut out = Vec::with_capacity(src.len());
            out.extend(src.iter().skip(1).step_by(2));
            out.extend(src.iter().step_by(2));
            out
        };
        report("delta-in-time, byte planes", split(&dt));
        report("plain, byte planes", split(&flat));

        // Delta across frequency instead — neighbouring bars are correlated too.
        let mut df = Vec::with_capacity(flat.len());
        for f in 0..frames {
            for b in 0..bars {
                let cur = vals[f * bars + b] as i32;
                let prev = if b == 0 { 0 } else { vals[f * bars + b - 1] as i32 };
                df.extend_from_slice(&zig(cur - prev).to_le_bytes());
            }
        }
        report("delta-in-freq, byte planes", split(&df));

        // How much of the cost is the noisy low byte? Upper bound on any
        // lossless scheme that keeps all 16 bits.
        let hi: Vec<u8> = vals.iter().map(|v| (v >> 8) as u8).collect();
        let mut hi_dt = Vec::with_capacity(hi.len());
        for f in 0..frames {
            for b in 0..bars {
                let cur = hi[f * bars + b] as i32;
                let prev = if f == 0 { 0 } else { hi[(f - 1) * bars + b] as i32 };
                hi_dt.push(((cur - prev) as i8) as u8);
            }
        }
        let hi_c = lz4_flex::compress_prepend_size(&hi_dt).len();
        println!("\n  high byte alone, delta-in-time: {:.1} MB.", mb(hi_c));
        println!("  So a lossless scheme cannot beat ~{:.0} MB: the low byte is {:.0} MB \
                  of uniform noise and no coder compresses that.\n",
                 mb(hi_c) + mb(vals.len()), mb(vals.len()));

        // Bit depth is therefore the only real lever. The stored value spans an
        // 80 dB display range, so a level is 80/2^bits dB — compare that with
        // what one screen pixel is worth.
        println!("{:<34} {:>7}  {:>9}  {:>10}", "bit depth", "size", "dB/level", "vs 4K px");
        for bits in [16u32, 14, 12, 11, 10, 8] {
            let levels = 1u32 << bits;
            let q: Vec<u16> = vals
                .iter()
                .map(|&v| {
                    let step = 65_536 / levels;
                    (v / step as u16) * step as u16
                })
                .collect();
            let mut d = Vec::with_capacity(q.len() * 2);
            for f in 0..frames {
                for b in 0..bars {
                    let cur = q[f * bars + b] as i32;
                    let prev = if f == 0 { 0 } else { q[(f - 1) * bars + b] as i32 };
                    d.extend_from_slice(&zig(cur - prev).to_le_bytes());
                }
            }
            let c = lz4_flex::compress_prepend_size(&split(&d)).len();
            let db_per_level = 80.0 / levels as f64;
            // A 4K panel is 2160 px tall; the plot gets most of that.
            let db_per_px = 80.0 / 2160.0;
            println!("{:<34} {:>7.1} MB {:>9.4} {:>9.1}x",
                     format!("  {bits}-bit, delta, planes"), mb(c), db_per_level,
                     db_per_level / db_per_px);
        }
        println!("\n  A level coarser than 1.0x a 4K pixel is visible banding; finer is not.");
    }

    /// v3 must survive a round trip to within its own quantisation step, and v2
    /// files must still load — there are gigabytes of them on real machines.
    #[test]
    fn cache_round_trips_and_reads_v2() {
        let n_bars = 64;
        let frames: Vec<Vec<f32>> = (0..40)
            .map(|f| {
                (0..n_bars)
                    .map(|b| {
                        let x = (f as f32 * 0.13 + b as f32 * 0.017).sin() * 0.5 + 0.5;
                        x.clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect();

        let p = std::env::temp_dir().join("moosik_cache_v3_roundtrip.spectrumcache");
        let _ = std::fs::remove_file(&p);
        save_cache(&p, &frames);
        let back = load_cache(&p, n_bars).expect("v3 failed to load");
        assert_eq!(back.len(), frames.len());
        let worst = frames.iter().flatten().zip(back.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // One u16 step is 1/65535; rounding can cost half of one either way.
        assert!(worst <= 1.0 / 65535.0 + 1e-7, "round trip lost {worst}");

        // Hand-build a v2 file and check the reader still accepts it.
        let mut payload = Vec::new();
        for row in &frames {
            for &v in row {
                payload.extend_from_slice(&(((v * 65535.0).round()) as u16).to_le_bytes());
            }
        }
        let comp = lz4_flex::compress_prepend_size(&payload);
        let mut blob = Vec::new();
        blob.extend_from_slice(&CACHE_MAGIC.to_le_bytes());
        blob.extend_from_slice(&(frames.len() as u32).to_le_bytes());
        blob.extend_from_slice(&(n_bars as u32).to_le_bytes());
        blob.extend_from_slice(&comp);
        let p2 = std::env::temp_dir().join("moosik_cache_v2_compat.spectrumcache");
        std::fs::write(&p2, &blob).unwrap();
        let old = load_cache(&p2, n_bars).expect("v2 file no longer loads");
        let worst_v2 = frames.iter().flatten().zip(old.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst_v2 <= 1.0 / 65535.0 + 1e-7);

        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(&p2);
    }

    /// A cache written for a different bar count must be rejected, not
    /// reinterpreted — the v3 reader indexes two planes and would read garbage.
    #[test]
    fn cache_rejects_mismatched_bar_count() {
        let frames: Vec<Vec<f32>> = (0..8).map(|_| vec![0.5f32; 32]).collect();
        let p = std::env::temp_dir().join("moosik_cache_bars_mismatch.spectrumcache");
        let _ = std::fs::remove_file(&p);
        save_cache(&p, &frames);
        assert!(load_cache(&p, 64).is_none(), "accepted a 32-bar cache as 64 bars");
        assert!(load_cache(&p, 32).is_some());
        let _ = std::fs::remove_file(&p);
    }

    /// Eviction must free enough, take the oldest first, and touch nothing that
    /// is not a cache file.
    #[test]
    fn eviction_removes_oldest_until_under_budget() {
        let dir = std::env::temp_dir().join("moosik_evict_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // 5 files of 1000 bytes, written oldest-first.
        let mut paths = Vec::new();
        for i in 0..5 {
            let p = dir.join(format!("f{i}.spectrumcache"));
            std::fs::write(&p, vec![0u8; 1000]).unwrap();
            // Space the timestamps so the sort is unambiguous.
            std::thread::sleep(std::time::Duration::from_millis(20));
            paths.push(p);
        }
        let innocent = dir.join("notes.txt");
        std::fs::write(&innocent, vec![0u8; 4000]).unwrap();

        // 5000 bytes against a 3000 budget: drop the two oldest, landing exactly
        // on the limit.
        let (removed, freed) = evict_in_dir(&dir, 3000, None);
        assert_eq!(removed, 2, "removed {removed}, expected 2");
        assert_eq!(freed, 2000);
        assert!(!paths[0].exists() && !paths[1].exists(), "oldest not evicted first");
        assert!(paths[2].exists() && paths[3].exists() && paths[4].exists());
        assert!(innocent.exists(), "deleted a file that was not a cache");

        // Already under budget: nothing happens.
        assert_eq!(evict_in_dir(&dir, 10_000, None), (0, 0));

        // `keep` is spared even when it is the oldest.
        let (removed, _) = evict_in_dir(&dir, 1000, Some(&paths[2]));
        assert!(paths[2].exists(), "kept file was evicted anyway");
        assert!(removed >= 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn eta_wording_is_coarse() {
        assert_eq!(fmt_eta(0), "a few seconds");
        assert_eq!(fmt_eta(45), "45s");
        assert_eq!(fmt_eta(150), "3m");
        assert_eq!(fmt_eta(3600), "1h");
        assert_eq!(fmt_eta(4500), "1h 15m");
    }
}

#[cfg(test)]
mod superlet_pipeline_tests {
    use super::*;

    /// Minimal 16-bit mono WAV. Hand-rolled rather than pulling in a writer
    /// crate for four tests.
    fn write_wav(path: &PathBuf, samples: &[f32], sr: u32) {
        let data: Vec<u8> = samples
            .iter()
            .flat_map(|&s| ((s.clamp(-1.0, 1.0) * 32_767.0) as i16).to_le_bytes())
            .collect();
        let mut out = Vec::with_capacity(44 + data.len());
        out.extend(b"RIFF");
        out.extend(((36 + data.len()) as u32).to_le_bytes());
        out.extend(b"WAVEfmt ");
        out.extend(16u32.to_le_bytes());
        out.extend(1u16.to_le_bytes());   // PCM
        out.extend(1u16.to_le_bytes());   // mono
        out.extend(sr.to_le_bytes());
        out.extend((sr * 2).to_le_bytes());
        out.extend(2u16.to_le_bytes());
        out.extend(16u16.to_le_bytes());
        out.extend(b"data");
        out.extend((data.len() as u32).to_le_bytes());
        out.extend(data);
        std::fs::write(path, out).expect("write wav");
    }

    struct Fixture { wav: PathBuf, cache: PathBuf }

    impl Fixture {
        fn new(tag: &str, freq: f32, secs: f32, sr: u32) -> Self {
            let dir = std::env::temp_dir();
            let wav = dir.join(format!("moosik_aslt_{tag}.wav"));
            let cache = dir.join(format!("moosik_aslt_{tag}.spectrumcache"));
            let n = (secs * sr as f32) as usize;
            let sig: Vec<f32> = (0..n)
                .map(|i| 0.8 * (std::f32::consts::TAU * freq * i as f32 / sr as f32).sin())
                .collect();
            write_wav(&wav, &sig, sr);
            let _ = std::fs::remove_file(&cache);
            Self { wav, cache }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.wav);
            let _ = std::fs::remove_file(&self.cache);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run(f: &Fixture, n_bars: usize, fps: f32, abort: &Arc<AtomicBool>) -> PreMessage {
        preprocess_file(
            &f.wav, &f.cache, 44_100, n_bars,
            &Arc::new(AtomicUsize::new(0)),
            8192, 16, 0.875, &WindowFn::Hann, 200.0, 5000.0,
            &InterpolationMode::None, &BarMappingMode::Superlet,
            crate::dsd::decimate::DEFAULT_ANALYSIS_RATE,
            &aslt::AsltPreset::Fast.config(), fps, abort,
            &Arc::new(AtomicUsize::new(usize::MAX)),
        )
    }

    /// Decode → loudness pass → superlet → dB → cache, on a real file.
    #[test]
    fn end_to_end_produces_a_peak_at_the_right_bar() {
        let f = Fixture::new("tone", 1000.0, 1.0, 44_100);
        let n_bars = 64;
        let abort = Arc::new(AtomicBool::new(false));
        let PreMessage::Done { frames, frame_rate, .. } = run(&f, n_bars, 60.0, &abort) else {
            panic!("expected Done");
        };

        // Frame rate must come from the requested fps, not from window/overlap.
        assert!((frame_rate - 60.0).abs() < 1.0, "frame rate was {frame_rate}");
        assert!(frames.len() > 30, "only {} frames", frames.len());
        assert!(frames.iter().all(|r| r.len() == n_bars));

        let mid = &frames[frames.len() / 2];
        let (peak_bar, &peak) = mid.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        let peak_hz = aslt::bar_center_freq(peak_bar, n_bars, 200.0, 5000.0);
        assert!((peak_hz - 1000.0).abs() < 120.0, "peak landed at {peak_hz} Hz");
        assert!(peak > 0.5, "peak only reached {peak} of full scale");

        // Everything stays inside the normalised display range.
        assert!(frames.iter().flatten().all(|&v| (0.0..=1.0).contains(&v)));
        assert!(f.cache.exists(), "cache was not written");
    }

    /// dB calibration: a −20 dBFS tone must sit ~20 dB below a 0 dBFS one on the
    /// normalised 80 dB scale, i.e. a quarter of the range.
    #[test]
    fn amplitude_maps_to_the_expected_db_offset() {
        let sr = 44_100u32;
        let dir = std::env::temp_dir();
        let mut levels = Vec::new();
        for (tag, amp) in [("loud", 1.0f32), ("quiet", 0.1f32)] {
            let wav = dir.join(format!("moosik_aslt_lvl_{tag}.wav"));
            let cache = dir.join(format!("moosik_aslt_lvl_{tag}.spectrumcache"));
            let n = sr as usize;
            let sig: Vec<f32> = (0..n)
                .map(|i| amp * (std::f32::consts::TAU * 1000.0 * i as f32 / sr as f32).sin())
                .collect();
            write_wav(&wav, &sig, sr);
            let _ = std::fs::remove_file(&cache);
            let f = Fixture { wav, cache };
            let abort = Arc::new(AtomicBool::new(false));
            let PreMessage::Done { frames, .. } = run(&f, 64, 60.0, &abort) else {
                panic!("expected Done");
            };
            let mid = &frames[frames.len() / 2];
            levels.push(mid.iter().cloned().fold(0.0f32, f32::max));
        }
        // 20 dB out of the 80 dB display range = 0.25 of full scale.
        let delta = levels[0] - levels[1];
        assert!((delta - 0.25).abs() < 0.06, "expected ~0.25 drop, got {delta} ({levels:?})");
    }

    /// Abort must stop the run and leave no cache behind — a partial cache would
    /// be reloaded later as though it were a complete analysis.
    #[test]
    fn abort_yields_no_cache() {
        let f = Fixture::new("abort", 1000.0, 1.0, 44_100);
        let abort = Arc::new(AtomicBool::new(true));
        assert!(matches!(run(&f, 64, 60.0, &abort), PreMessage::Aborted));
        assert!(!f.cache.exists(), "aborted run still wrote a cache");
    }
}

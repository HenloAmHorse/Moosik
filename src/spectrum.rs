pub mod eq;
pub mod art;

use egui::{Color32, Pos2, Rect, Shape, Stroke};
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
pub const DEFAULT_MIN_FREQ: f32 = 20.0;
pub const DEFAULT_MAX_FREQ: f32 = 24_000.0;
const WATERFALL_ROWS: usize = 120;
// ---------------------------------------------------------------------------
// New public enums: loudness mode, window function
// ---------------------------------------------------------------------------
#[derive(Clone, PartialEq, Debug)]
pub enum LoudnessMode {
    /// Raw dB magnitude — no psychoacoustic correction.
    Flat,
    /// ISO 226:2003 equal-loudness weighting at 40 phon.
    EqualLoudness,
}

#[derive(Clone, PartialEq, Debug)]
pub enum WindowFn { Hann, Hamming, Blackman, FlatTop }

/// Sub-bin interpolation method used when mapping FFT bins to spectrum bars.
#[derive(Clone, PartialEq, Debug)]
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

#[derive(Clone, PartialEq, Debug)]
pub enum BarMappingMode {
    /// Flat overlap average — equal weight to all bins in range.
    FlatOverlap,
    /// Gaussian-weighted average — bins near bar centre frequency weighted more heavily.
    Gaussian,
    /// Constant-Q transform — Hann kernel whose bandwidth scales with frequency (f/Q).
    /// Each bar has the same relative frequency resolution regardless of pitch.
    Cqt,
}

// ---------------------------------------------------------------------------
// Peak hold
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Debug)]
pub enum PeakDecayMode {
    /// Constant fall speed.
    Linear,
    /// Accelerates as it falls — feels physical.
    Gravity,
    /// Peak fades in place rather than falling.
    FadeOut,
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
pub fn compute_eq_weights(n_bars: usize, min_freq: f32, max_freq: f32) -> Vec<f32> {
    let ref_spl = iso226_spl_40phon(1000.0); // ≈ 40.0 by definition
    let log_min = min_freq.log10();
    let log_max = max_freq.log10();
    (0..n_bars)
        .map(|b| {
            let t = (b as f32 + 0.5) / n_bars as f32;
            let freq = 10_f32.powf(log_min + t * (log_max - log_min));
            let freq_clamped = freq.clamp(20.0, 12_500.0);
            ref_spl - iso226_spl_40phon(freq_clamped)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Chromagram, key detection, chord detection, BPM
// ---------------------------------------------------------------------------

const NOTE_NAMES: &[&str] = &["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];

/// Krumhansl-Schmuckler key profiles (major / minor).
const KS_MAJOR: [f32; 12] = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88];
const KS_MINOR: [f32; 12] = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17];

/// Soft triad templates (root = index 0).
/// Weights reflect perceptual salience: root > 5th > 3rd (harmonics taper off).
const CHORD_MAJ_T: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0];
const CHORD_MIN_T: [f32; 12] = [1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0];

/// Convert FFT half-spectrum → 12-bin chromagram (pitch-class energy sums).
///
/// Key improvements over the naive approach:
/// - Frequency range 110 Hz–4186 Hz (A2–C8): covers chord fundamentals and
///   their first few harmonics without being swamped by sub-bass or high noise.
/// - Log-magnitude compression: `log2(1 + v*scale)` equalises the energy
///   contribution across octaves; without this, bass bins dominate.
/// - Returns unnormalised values so callers can accumulate across multiple
///   frames (energy-weighted) before normalising at match time.
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

/// Match a chroma frame to the best major/minor triad.
/// Returns 0–11 = C..B major, 12–23 = Cm..Bm, 255 = too quiet.
/// L1-normalises on the stack — zero heap allocation.
fn match_chord(chroma: &[f32; 12]) -> u8 {
    let sum: f32 = chroma.iter().sum();
    if sum < 1e-6 { return 255; }
    let cn: [f32; 12] = std::array::from_fn(|i| chroma[i] / sum);
    let mut best_score = -2.0f32;
    let mut best = 255u8;
    for root in 0..12usize {
        let maj: [f32; 12] = std::array::from_fn(|i| CHORD_MAJ_T[(i + 12 - root) % 12]);
        let min: [f32; 12] = std::array::from_fn(|i| CHORD_MIN_T[(i + 12 - root) % 12]);
        let ms = cosine_sim(&cn, &maj);
        let mn = cosine_sim(&cn, &min);
        if ms > best_score { best_score = ms; best = root as u8; }
        if mn > best_score { best_score = mn; best = (root + 12) as u8; }
    }
    best
}

/// Human-readable chord name. 255 = silence/unknown.
pub fn chord_name(idx: u8) -> &'static str {
    match idx {
        0  => "C",    1  => "C#",   2  => "D",    3  => "D#",
        4  => "E",    5  => "F",    6  => "F#",   7  => "G",
        8  => "G#",   9  => "A",   10  => "A#",  11  => "B",
        12 => "Cm",  13 => "C#m",  14 => "Dm",   15 => "D#m",
        16 => "Em",  17 => "Fm",   18 => "F#m",  19 => "Gm",
        20 => "G#m", 21 => "Am",   22 => "A#m",  23 => "Bm",
        _  => "—",
    }
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

#[derive(Clone, PartialEq, Debug)]
pub enum SpectrumMode {
    RealTime,
    PreProcess,
}

#[derive(Clone, PartialEq, Debug)]
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

/// Stereo sample pairs buffer (shared audio thread → UI thread).
pub type StereoBuf = Arc<Mutex<Vec<[f32; 2]>>>;
pub fn new_stereo_buf() -> StereoBuf {
    Arc::new(Mutex::new(Vec::with_capacity(8192)))
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
    pub chord_timeline: Vec<u8>,   // chord index per step (see chord_name()); 255 = silent
    pub chord_step_secs: f32,      // seconds per chord step
    pub loudness_history: Vec<f32>,// per-second integrated LUFS
}

// ---------------------------------------------------------------------------
// SpectrumSource — thin rodio Source wrapper that taps samples into SampleBuf
// ---------------------------------------------------------------------------

/// Samples are batched on the audio thread and flushed in bulk every
/// BATCH_SIZE mono frames — reduces mutex lock attempts from ~88k/sec
/// (per-sample) down to ~172/sec, cutting audio-thread overhead 512×.
const BATCH_SIZE: usize = 512;

pub struct SpectrumSource<S>
where
    S: rodio::Source + Send + 'static,
    S::Item: SampleToF32 + rodio::Sample,
{
    inner: S,
    buf: SampleBuf,
    stereo_buf: StereoBuf,
    channels: u16,
    ch_idx: u16,
    pending_l: f32,
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
        Self {
            inner, buf, stereo_buf, channels, ch_idx: 0, pending_l: 0.0,
            sample_batch: Vec::with_capacity(BATCH_SIZE),
            stereo_batch: Vec::with_capacity(BATCH_SIZE / 2),
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

        // Accumulate stereo pairs locally
        if self.channels == 2 {
            if self.ch_idx == 0 {
                self.pending_l = f;
            } else {
                self.stereo_batch.push([self.pending_l, f]);
            }
            self.ch_idx = (self.ch_idx + 1) % 2;
        }
        self.sample_batch.push(f);

        // Flush to shared buffers once per batch — one lock per 512 samples
        if self.sample_batch.len() >= BATCH_SIZE {
            if let Ok(mut v) = self.buf.try_lock() {
                v.extend_from_slice(&self.sample_batch);
                const CAP: usize = DEFAULT_FFT_SIZE * 4;
                if v.len() > CAP { let d = v.len() - CAP; v.drain(0..d); }
            }
            self.sample_batch.clear();

            if !self.stereo_batch.is_empty() {
                if let Ok(mut v) = self.stereo_buf.try_lock() {
                    v.extend_from_slice(&self.stereo_batch);
                    const CAP: usize = 8192;
                    if v.len() > CAP { let d = v.len() - CAP; v.drain(0..d); }
                }
                self.stereo_batch.clear();
            }
        }
        Some(s)
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
    pub sample_rate: u32,
    pub bar_count: usize,
    // Configurable FFT parameters
    pub fft_size: usize,
    pub window_fn: WindowFn,
    pub smoothing: f32,   // 0.0 = off, 0.9 = very smooth
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
            sample_rate: 44_100,
            bar_count: DEFAULT_BAR_COUNT,
            fft_size,
            window_fn: WindowFn::Hann,
            smoothing: 0.75,
            min_freq: DEFAULT_MIN_FREQ,
            max_freq: DEFAULT_MAX_FREQ,
            eq_weights: Vec::new(),
            interp_mode: InterpolationMode::None,
            pad_factor: DEFAULT_PAD,
            overlap: 0.875,
            bar_mapping: BarMappingMode::Cqt,
            pending_waveform: None,
            pending_analysis: None,
            pre_frames: Vec::new(),
            pre_frame_rate: 30.0,
            pre_receiver: None,
            is_analyzing: Arc::new(AtomicBool::new(false)),
            analysis_progress: Arc::new(AtomicUsize::new(0)),
            waterfall: Vec::new(),
            waterfall_dirty: false,
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
                let mag = if self.bar_mapping == BarMappingMode::Cqt {
                    // CQT: centre-frequency Hann kernel with constant-Q bandwidth
                    let tc  = (bar as f32 + 0.5) / n_bars as f32;
                    let f_c = 10_f32.powf(log_min + tc * (log_max - log_min));
                    let bc  = (f_c * padded_size as f32 / sr as f32).clamp(1.0, half as f32 - 1.0);
                    cqt_kernel(&norms, bc, cqt_q, half)
                } else {
                    let t0 = bar as f32 / n_bars as f32;
                    let t1 = (bar + 1) as f32 / n_bars as f32;
                    let freq_lo = 10_f32.powf(log_min + t0 * (log_max - log_min));
                    let freq_hi = 10_f32.powf(log_min + t1 * (log_max - log_min));
                    let fbin_lo = (freq_lo * padded_size as f32 / sr as f32).max(1.0);
                    let fbin_hi = (freq_hi * padded_size as f32 / sr as f32)
                        .max(fbin_lo + 0.001).min(half as f32 - 0.001);

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
                                BarMappingMode::Gaussian | BarMappingMode::Cqt => {
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

    pub fn process_realtime(&mut self) {
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
        let alpha = self.smoothing;
        for (s, m) in self.smoothed.iter_mut().zip(new_bars.iter()) {
            *s = *s * alpha + m * (1.0 - alpha);
        }
        self.magnitudes = self.smoothed.clone();
        self.push_waterfall(self.magnitudes.clone());
    }

    pub fn tick_pre(&mut self, elapsed: f64) {
        // Poll for completed background work
        let mut ready: Option<(Vec<Vec<f32>>, f64)> = None;
        if let Some(ref rx) = self.pre_receiver && let Ok(msg) = rx.try_recv() {
            match msg {
                PreMessage::Done { frames, frame_rate, waveform, analysis } => {
                    ready = Some((frames, frame_rate));
                    self.pending_waveform = Some(waveform);
                    self.pending_analysis = Some(analysis);
                }
                PreMessage::Error(_) => {
                    // is_analyzing already cleared by ClearOnDrop in thread
                }
            }
        }
        if let Some((frames, rate)) = ready {
            self.pre_frames = frames;
            self.pre_frame_rate = rate;
            self.pre_receiver = None;
            // Flag already cleared by the thread's ClearOnDrop guard
        }

        if !self.pre_frames.is_empty() {
            let frame = ((elapsed * self.pre_frame_rate) as usize)
                .min(self.pre_frames.len().saturating_sub(1));
            let mags = self.pre_frames[frame].clone();
            // Guard: if bar_count changed mid-stream, resize smooth buffer
            if self.smoothed.len() != mags.len() {
                self.smoothed = vec![0.0; mags.len()];
            }
            for (s, m) in self.smoothed.iter_mut().zip(mags.iter()) {
                *s = *s * 0.5 + m * 0.5;
            }
            self.magnitudes = self.smoothed.clone();
            self.push_waterfall(self.magnitudes.clone());
        }
    }

    pub fn start_preprocess(&mut self, path: PathBuf) {
        // Guard: refuse to spawn a second thread if one is already running.
        if self.is_analyzing.load(Ordering::Relaxed) {
            return;
        }
        let n_bars = self.bar_count;
        let cache = cache_path_for(
            &path, n_bars, self.fft_size, self.pad_factor, self.overlap,
            &self.window_fn, self.min_freq, self.max_freq, &self.bar_mapping, &self.interp_mode,
        );
        if let Some(frames) = load_cache(&cache, n_bars) {
            let hop = ((self.fft_size as f32 * (1.0 - self.overlap)).round() as usize).max(1);
            let rate = self.sample_rate as f64 / hop as f64;
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
            self.pre_frames = frames;
            self.pre_frame_rate = rate;
            return;
        }
        self.pre_frames.clear();
        self.analysis_progress.store(0, Ordering::Relaxed);
        self.is_analyzing.store(true, Ordering::Relaxed);
        let (tx, rx) = std::sync::mpsc::channel();
        self.pre_receiver = Some(rx);
        let sr = self.sample_rate;
        let fft_size    = self.fft_size;
        let pad_factor  = self.pad_factor;
        let window_fn   = self.window_fn.clone();
        let min_freq    = self.min_freq;
        let max_freq    = self.max_freq;
        let eq_weights  = self.eq_weights.clone();
        let interp_mode = self.interp_mode.clone();
        let overlap     = self.overlap;
        let bar_mapping = self.bar_mapping.clone();
        let flag = Arc::clone(&self.is_analyzing);
        let progress = Arc::clone(&self.analysis_progress);
        std::thread::spawn(move || {
            struct ClearOnDrop(Arc<AtomicBool>);
            impl Drop for ClearOnDrop {
                fn drop(&mut self) { self.0.store(false, Ordering::Relaxed); }
            }
            let _guard = ClearOnDrop(flag);
            let result = preprocess_file(
                &path, &cache, sr, n_bars, &progress,
                fft_size, pad_factor, overlap, &window_fn, min_freq, max_freq, &eq_weights, &interp_mode, &bar_mapping,
            );
            let _ = tx.send(result);
        });
    }

    fn push_waterfall(&mut self, row: Vec<f32>) {
        if !self.waterfall_enabled { return; }
        self.waterfall.push(row);
        if self.waterfall.len() > WATERFALL_ROWS {
            self.waterfall.remove(0);
        }
        self.waterfall_dirty = true;
    }

    /// Resize to a new bar count and clear all derived state.
    /// Drops the receiver so any in-flight thread's send silently fails.
    pub fn set_bar_count(&mut self, n: usize) {
        if n == self.bar_count { return; }
        self.bar_count = n;
        self.magnitudes = vec![0.0; n];
        self.smoothed = vec![0.0; n];
        self.eq_weights.clear();
        self.waterfall.clear();
        self.waterfall_dirty = false;
        self.pre_frames.clear();
        self.pre_receiver = None;
        self.is_analyzing.store(false, Ordering::Relaxed);
        self.analysis_progress.store(0, Ordering::Relaxed);
    }

    pub fn reset(&mut self) {
        let n = self.bar_count;
        self.magnitudes = vec![0.0; n];
        self.smoothed = vec![0.0; n];
        self.eq_weights.clear();
        self.waterfall.clear();
        self.waterfall_dirty = false;
        self.pre_frames.clear();
        self.pre_receiver = None;
        // Replace the Arcs rather than just storing false/0 into them.
        // Any lingering old-thread ClearOnDrop guard holds a clone of the
        // *previous* Arc; when it fires it writes to that abandoned Arc
        // instead of clobbering the flag for the new analysis we're about
        // to start.
        self.is_analyzing = Arc::new(AtomicBool::new(false));
        self.analysis_progress = Arc::new(AtomicUsize::new(0));
        self.pending_waveform = None;
        self.pending_analysis = None;
        if let Ok(mut b) = self.sample_buf.lock() {
            b.clear();
        }
    }

    /// Poll the background pre-process channel and store frames if ready,
    /// without updating magnitudes. Called by SpectrumWindow before the
    /// is_playing guard so results arrive even when paused.
    pub fn try_receive_frames(&mut self) {
        if let Some(ref rx) = self.pre_receiver && let Ok(msg) = rx.try_recv() {
            match msg {
                PreMessage::Done { frames, frame_rate, waveform, analysis } => {
                    self.pre_frames = frames;
                    self.pre_frame_rate = frame_rate;
                    self.pending_waveform = Some(waveform);
                    self.pending_analysis = Some(analysis);
                }
                PreMessage::Error(_) => {}
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

fn home_dir() -> PathBuf {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

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
) -> PathBuf {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    path.hash(&mut h);
    let hash = h.finish();
    let overlap_k  = (overlap * 1000.0) as u32;
    let window_id: u8 = match window_fn {
        WindowFn::Hann => 0, WindowFn::Hamming => 1,
        WindowFn::Blackman => 2, WindowFn::FlatTop => 3,
    };
    let mapping_id: u8 = match bar_mapping {
        BarMappingMode::FlatOverlap => 0, BarMappingMode::Gaussian => 1, BarMappingMode::Cqt => 2,
    };
    let interp_id: u8 = match interp_mode {
        InterpolationMode::None => 0, InterpolationMode::Linear => 1,
        InterpolationMode::CatmullRom => 2, InterpolationMode::Pchip => 3,
        InterpolationMode::Akima => 4, InterpolationMode::Lanczos => 5,
    };
    let min_hz = min_freq.round() as u32;
    let max_hz = max_freq.round() as u32;
    home_dir()
        .join(".moosik")
        .join("cache")
        .join(format!(
            "{:016x}_b{}_f{}_w{}_p{}_o{}_n{}_x{}_m{}_i{}.spectrumcache",
            hash, n_bars, fft_size, window_id, pad_factor, overlap_k, min_hz, max_hz, mapping_id, interp_id,
        ))
}

fn load_cache(cache_path: &PathBuf, n_bars: usize) -> Option<Vec<Vec<f32>>> {
    use std::io::Read;
    let mut file = std::fs::File::open(cache_path).ok()?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).ok()?;
    if data.len() < 8 {
        return None;
    }
    let num_frames = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
    let num_bars  = u32::from_le_bytes(data[4..8].try_into().ok()?) as usize;

    // Sanity caps: reject obviously bogus / crafted headers before arithmetic.
    if num_bars != n_bars || num_bars > MAX_BAR_COUNT || num_frames > 500_000 {
        return None;
    }

    // Use saturating arithmetic so a huge header can't overflow `expected`.
    let payload = num_frames.saturating_mul(num_bars).saturating_mul(4);
    let expected = 8usize.saturating_add(payload);
    if data.len() < expected {
        return None;
    }

    // Decode frames — propagate any conversion failure as None instead of panicking.
    let frames: Option<Vec<Vec<f32>>> = (0..num_frames)
        .map(|f| {
            (0..num_bars)
                .map(|b| {
                    let off = 8 + (f * num_bars + b) * 4;
                    let bytes: [u8; 4] = data[off..off + 4].try_into().ok()?;
                    Some(f32::from_le_bytes(bytes))
                })
                .collect::<Option<Vec<f32>>>()
        })
        .collect();
    frames
}

fn save_cache(cache_path: &PathBuf, frames: &[Vec<f32>]) {
    use std::io::Write;
    if let Some(parent) = cache_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut data: Vec<u8> = Vec::new();
    let n_bars = frames.first().map(|f| f.len()).unwrap_or(0);
    data.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    data.extend_from_slice(&(n_bars as u32).to_le_bytes());
    for frame in frames {
        for &v in frame {
            data.extend_from_slice(&v.to_le_bytes());
        }
    }
    if let Ok(mut f) = std::fs::File::create(cache_path) && f.write_all(&data).is_err() {
        let _ = std::fs::remove_file(cache_path);
    }
}

// ---------------------------------------------------------------------------
// Background pre-processing (uses rodio Decoder directly)
// ---------------------------------------------------------------------------

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
    eq_weights: &[f32],
    interp_mode: &InterpolationMode,
    bar_mapping: &BarMappingMode,
) -> PreMessage {
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

    let sample_rate = decoder.sample_rate();
    let ch = decoder.channels() as usize;

    let hop = ((fft_size as f32 * (1.0 - overlap)).round() as usize).max(1);
    let padded_size = fft_size * pad_factor;
    let window = make_window(fft_size, window_fn);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(padded_size);
    let half = padded_size / 2;
    let scale = fft_size as f32; // normalize by window length, not padded length
    let log_min = min_freq.log10();
    let log_max = (sample_rate as f32 / 2.0).min(max_freq).log10();

    // Estimate total mono samples for progress reporting (best-effort).
    let total_hint = decoder.total_duration()
        .map(|d| (d.as_secs_f64() * sample_rate as f64) as usize)
        .unwrap_or(0);

    // ── Phase 1: decode all mono samples sequentially (0–49 %) ──────────────
    let ch = ch.max(1);
    let mut all_mono: Vec<f32> = Vec::with_capacity(total_hint.max(1));
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
        all_mono.push(sum / got as f32);
        if total_hint > 0 {
            progress.store((all_mono.len() * 49 / total_hint).min(49), Ordering::Relaxed);
        }
    }

    if all_mono.len() < fft_size {
        return PreMessage::Error("audio too short for analysis".into());
    }
    progress.store(50, Ordering::Relaxed);

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
    let chord_hop_frames = ((1.5 * sample_rate as f32) / CHROMA_HOP as f32).ceil() as usize;
    let chord_hop_frames = chord_hop_frames.max(1);
    let chord_step_secs  = chord_hop_frames as f32 * CHROMA_HOP as f32 / sample_rate as f32;
    let mut chord_acc    = [0.0f32; 12];
    let mut chord_frame_count = 0usize;
    let mut chord_timeline: Vec<u8> = Vec::new();
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

    for (idx, &s) in all_mono.iter().enumerate() {
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
            for (a, &b) in chord_acc.iter_mut().zip(&fc) { *a += b; }
            chord_frame_count += 1;
            if chord_frame_count >= chord_hop_frames {
                chord_timeline.push(match_chord(&chord_acc));
                chord_acc = [0.0; 12];
                chord_frame_count = 0;
            }
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
    if chord_frame_count > 0 { chord_timeline.push(match_chord(&chord_acc)); }

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
        chord_timeline, chord_step_secs, loudness_history,
    };

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
                    let mag = if *bar_mapping == BarMappingMode::Cqt {
                        let tc  = (bar as f32 + 0.5) / n_bars as f32;
                        let f_c = 10_f32.powf(log_min + tc * (log_max - log_min));
                        let bc  = (f_c * padded_size as f32 / sample_rate as f32).clamp(1.0, half as f32 - 1.0);
                        cqt_kernel(&norms, bc, cqt_q_pp, half)
                    } else {
                        let t0 = bar as f32 / n_bars as f32;
                        let t1 = (bar + 1) as f32 / n_bars as f32;
                        let freq_lo = 10_f32.powf(log_min + t0 * (log_max - log_min));
                        let freq_hi = 10_f32.powf(log_min + t1 * (log_max - log_min));
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
                                    BarMappingMode::Gaussian | BarMappingMode::Cqt => {
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
                    let db = if let Some(&w) = eq_weights.get(bar) { raw_db + w } else { raw_db };
                    ((db + 80.0) / 80.0).clamp(0.0, 1.0)
                })
                .collect();

            let c = done.fetch_add(1, Ordering::Relaxed) + 1;
            progress.store(50 + (c * 49 / nf).min(49), Ordering::Relaxed);
            bars
        })
        .collect();

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
    let log_min = min_freq.log10();
    let log_max = max_freq.log10();
    let t = (highest as f32 + 0.5) / n_bars as f32;
    let hz = 10_f32.powf(log_min + t * (log_max - log_min));
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
    fn push_frame(&mut self, norms: &[f32], sr: u32, fft_size: usize, min_freq: f32, max_freq: f32) {
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
            self.pixels[row * SPEC_W + self.col_head] = heat_color(v);
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
    plot_rect: Rect, sr: u32, min_freq: f32, max_freq: f32,
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
            0.0, bar_color(mag),
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
    let b1 = center.floor() as usize;
    let t  = center - b1 as f32;
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

fn bar_color(t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.2 {
        // silence → dark blue
        let s = t / 0.2;
        Color32::from_rgb(0, 0, (s * 140.0) as u8)
    } else if t < 0.4 {
        // dark blue → cyan
        let s = (t - 0.2) / 0.2;
        Color32::from_rgb(0, (s * 220.0) as u8, (140.0 + s * 115.0) as u8)
    } else if t < 0.6 {
        // cyan → yellow
        let s = (t - 0.4) / 0.2;
        Color32::from_rgb((s * 255.0) as u8, 220, (255.0 * (1.0 - s)) as u8)
    } else if t < 0.8 {
        // yellow → orange-red
        let s = (t - 0.6) / 0.2;
        Color32::from_rgb(255, (220.0 - s * 150.0) as u8, 0)
    } else {
        // orange-red → white-hot
        let s = (t - 0.8) / 0.2;
        Color32::from_rgb(255, (70.0 + s * 185.0) as u8, (s * 220.0) as u8)
    }
}

fn heat_color(t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.25 {
        let s = t / 0.25;
        Color32::from_rgb(0, 0, (s * 180.0) as u8)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        Color32::from_rgb(0, (s * 200.0) as u8, 180)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        Color32::from_rgb((s * 255.0) as u8, 200, (180.0 * (1.0 - s)) as u8)
    } else {
        let s = (t - 0.75) / 0.25;
        Color32::from_rgb(255, (200.0 + s * 55.0) as u8, (s * 80.0) as u8)
    }
}

// ---------------------------------------------------------------------------
// EQ overlay — response curve and draggable band nodes
// ---------------------------------------------------------------------------

/// EQ dB range shown on the overlay (±DB_RANGE maps to ±half plot height).
const EQ_DB_RANGE: f32 = 24.0;

fn eq_node_pos(band: &EqBand, plot_rect: Rect, min_freq: f32, max_freq: f32) -> egui::Pos2 {
    let log_min = min_freq.log10();
    let log_span = (max_freq.log10() - log_min).max(1e-6);
    let t = ((band.freq.log10() - log_min) / log_span).clamp(0.0, 1.0);
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
) {
    use egui::{Color32, Pos2, Stroke};

    let log_min  = min_freq.log10();
    let log_span = (max_freq.log10() - log_min).max(1e-6);
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
                let freq = 10f32.powf(log_min + t * log_span);
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
            let freq = 10f32.powf(log_min + t * log_span);
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
        let pos = eq_node_pos(band, plot_rect, min_freq, max_freq);
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

fn draw_bars(painter: &egui::Painter, mags: &[f32], rect: Rect, gap: f32) {
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
        let c  = bar_color(v);
        let base = mesh.vertices.len() as u32;
        mesh.colored_vertex(Pos2::new(x0, y0), c);  // top-left
        mesh.colored_vertex(Pos2::new(x1, y0), c);  // top-right
        mesh.colored_vertex(Pos2::new(x1, y1), c);  // bottom-right
        mesh.colored_vertex(Pos2::new(x0, y1), c);  // bottom-left
        mesh.indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }
    painter.add(Shape::mesh(mesh));
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

fn draw_filled(painter: &egui::Painter, mags: &[f32], rect: Rect) {
    let n = mags.len();
    if n < 2 { return; }
    let line_color = Color32::from_rgb(80, 210, 140);
    let fill_color = Color32::from_rgba_unmultiplied(40, 160, 100, 110);

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

fn waterfall_to_color_image(waterfall: &[Vec<f32>]) -> Option<egui::ColorImage> {
    let n = waterfall.first().map(|r| r.len()).unwrap_or(0);
    if n == 0 { return None; }
    let h = WATERFALL_ROWS;
    let mut pixels = vec![Color32::BLACK; n * h];
    for (row_idx, row) in waterfall.iter().enumerate() {
        // row 0 is most recent (rendered at bottom in original draw_waterfall).
        // In texture coordinates, row 0 is the top, so we flip.
        let tex_row = h.saturating_sub(1 + row_idx);
        let cols = row.len().min(n);
        for col in 0..cols {
            pixels[tex_row * n + col] = heat_color(row[col]);
        }
    }
    Some(egui::ColorImage { size: [n, h], pixels })
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

fn draw_freq_labels(painter: &egui::Painter, plot_rect: Rect, sample_rate: u32, min_freq: f32, max_freq: f32) {
    let nyquist = (sample_rate as f32 / 2.0).min(max_freq);
    let log_min = min_freq.log10();
    let log_max = nyquist.log10();
    let freqs: &[f32] = &[50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0];
    for &freq in freqs {
        if freq > nyquist { break; }
        let t = (freq.log10() - log_min) / (log_max - log_min);
        let x = plot_rect.left() + t * plot_rect.width();
        painter.line_segment(
            [Pos2::new(x, plot_rect.bottom()), Pos2::new(x, plot_rect.bottom() + 4.0)],
            Stroke::new(1.0, Color32::from_gray(60)),
        );
        let label = if freq >= 1000.0 { format!("{}k", (freq / 1000.0) as u32) }
                    else { format!("{}", freq as u32) };
        painter.text(
            Pos2::new(x, plot_rect.bottom() + 5.0),
            egui::Align2::CENTER_TOP,
            label,
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
// SpectrumWindow — the main public interface consumed by MoosikApp
// ---------------------------------------------------------------------------

pub struct SpectrumWindow {
    pub open: bool,
    pub mode: SpectrumMode,
    pub style: VizStyle,
    pub loudness_mode: LoudnessMode,
    pub analyzer: SpectrumAnalyzer,
    pub sample_buf: SampleBuf,
    pub bar_count: usize,
    pub bar_gap: f32,       // physical-pixel gap between bars (default 1)
    pub fft_size: usize,
    pub window_fn: WindowFn,
    pub smoothing: f32,
    pub min_freq: f32,
    pub max_freq: f32,
    current_path: Option<PathBuf>,
    status_msg: String,
    pub max_fps: f32,
    last_fft_time: Option<Instant>,
    current_fps: f32,
    pub waveform: Option<Vec<f32>>,
    waveform_rx: Option<std::sync::mpsc::Receiver<(Vec<f32>, TrackAnalysis)>>,
    pub spectral_ceiling: Option<SpectralCeiling>,
    pub stereo_buf: StereoBuf,
    pub track_analysis: Option<TrackAnalysis>,
    pub momentary_lufs: f32,
    pub correlation: f32,
    spectrogram: Spectrogram,
    spectrogram_texture: Option<egui::TextureHandle>,
    waterfall_texture: Option<egui::TextureHandle>,
    octave_bands: Vec<(f32, f32)>,
    octave_smoothed: Vec<f32>,
    pub show_chord: bool,
    pub current_chord: String,
    chroma_ema: [f32; 12],        // exponential moving average for live chord smoothing
    auto_fft_size: usize,         // last FFT size chosen automatically; 0 = user overrode it
    phasescope_frames: Vec<[f32; 2]>, // snapshot captured in tick() — no lock/clone in show()
    lufs_tick: u8, // simple counter for throttling LUFS computation
    pub interp_mode: InterpolationMode,
    pub pad_factor: usize,
    pub overlap: f32,
    pub bar_mapping: BarMappingMode,
    pub show_debug: bool,
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
    pub fn new() -> Self {
        let buf = new_sample_buf();
        let analyzer = SpectrumAnalyzer::new(Arc::clone(&buf));
        Self {
            open: false,
            mode: SpectrumMode::PreProcess,
            style: VizStyle::Bars,
            loudness_mode: LoudnessMode::Flat,
            analyzer,
            sample_buf: buf,
            bar_count: DEFAULT_BAR_COUNT,
            bar_gap: 1.0,
            fft_size: DEFAULT_FFT_SIZE,
            window_fn: WindowFn::Hann,
            smoothing: 0.75,
            min_freq: DEFAULT_MIN_FREQ,
            max_freq: DEFAULT_MAX_FREQ,
            current_path: None,
            status_msg: String::new(),
            max_fps: 60.0,
            last_fft_time: None,
            current_fps: 0.0,
            waveform: None,
            waveform_rx: None,
            spectral_ceiling: None,
            spectrogram: Spectrogram::new(),
            spectrogram_texture: None,
            waterfall_texture: None,
            octave_bands: Vec::new(),
            octave_smoothed: Vec::new(),
            stereo_buf: new_stereo_buf(),
            track_analysis: None,
            momentary_lufs: f32::NEG_INFINITY,
            correlation: 1.0,
            show_chord: false,
            current_chord: String::new(),
            chroma_ema: [0.0; 12],
            auto_fft_size: 0,
            phasescope_frames: Vec::new(),
            lufs_tick: 0,
            interp_mode: InterpolationMode::None,
            pad_factor: 16,
            overlap: 0.875,
            bar_mapping: BarMappingMode::Cqt,
            show_debug: false,
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
            art_settings: ArtSettingsStore::load(),
            current_art: None,
            peak_config: PeakHoldConfig::default(),
            peak_vals: Vec::new(),
            peak_hold_timers: Vec::new(),
            peak_velocities: Vec::new(),
            peak_alphas: Vec::new(),
        }
    }

    /// Recompute and push eq_weights (and other params) into the analyzer.
    fn sync_params(&mut self) {
        self.analyzer.min_freq = self.min_freq;
        self.analyzer.max_freq = self.max_freq;
        self.analyzer.smoothing = self.smoothing;
        self.analyzer.eq_weights = if self.loudness_mode == LoudnessMode::EqualLoudness {
            compute_eq_weights(self.bar_count, self.min_freq, self.max_freq)
        } else {
            Vec::new()
        };
    }

    /// After any quality-setting change, check whether a cache file already exists
    /// for the current path + new settings.  If yes, load it instantly and return true.
    /// If no, clear pre_frames and set `needs_reanalysis` so the UI can prompt the user.
    fn try_load_or_flag_reanalysis(&mut self) {
        let Some(path) = self.current_path.clone() else { return; };
        let cache = cache_path_for(
            &path, self.bar_count, self.fft_size, self.pad_factor, self.overlap,
            &self.window_fn, self.min_freq, self.max_freq, &self.bar_mapping, &self.interp_mode,
        );
        if let Some(frames) = load_cache(&cache, self.bar_count) {
            let hop = ((self.analyzer.fft_size as f32 * (1.0 - self.overlap)).round() as usize).max(1);
            let rate = self.analyzer.sample_rate as f64 / hop as f64;
            self.analyzer.pre_frames = frames;
            self.analyzer.pre_frame_rate = rate;
            self.spectral_ceiling = None; // recompute on next tick
            self.needs_reanalysis = false;
        } else {
            self.analyzer.pre_frames.clear();
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

        // Auto-scale FFT size to maintain the same ~85ms analysis window as
        // 8192 samples @ 96 kHz.  Picks the nearest supported power-of-two.
        let ideal = (8192.0_f32 * sample_rate as f32 / 96_000.0).round() as usize;
        let auto_fft = [1024usize, 2048, 4096, 8192, 16384]
            .iter()
            .min_by_key(|&&sz| (sz as i64 - ideal as i64).unsigned_abs())
            .copied()
            .unwrap_or(8192);
        self.fft_size = auto_fft;
        self.analyzer.fft_size = auto_fft;
        self.analyzer.rebuild_fft();
        self.auto_fft_size = auto_fft;

        self.current_path = Some(path.to_path_buf());
        // Auto-load the last-used (or default) preset for the new track.
        self.auto_load_preset_for(path);
        self.analyzer.reset();
        self.analyzer.sample_rate = sample_rate;
        self.analyzer.bar_count = self.bar_count;
        self.sync_params();
        self.spectral_ceiling = None;
        self.waveform = None;
        self.waveform_rx = None;
        self.spectrogram.clear();
        self.octave_bands.clear();
        self.octave_smoothed.clear();
        self.track_analysis = None;
        self.momentary_lufs = f32::NEG_INFINITY;
        self.correlation = 1.0;
        self.current_chord = String::new();
        self.chroma_ema = [0.0; 12];
        self.needs_reanalysis = false;
        if let Ok(mut v) = self.stereo_buf.lock() { v.clear(); }
        // Always preprocess — needed for spectral ceiling even in real-time mode
        self.analyzer.start_preprocess(path.to_path_buf());
        self.waveform_rx = None;
    }

    /// Call when playback stops or the player is stopped.
    pub fn on_stop(&mut self) {
        self.analyzer.reset();
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
        if self.spectral_ceiling.is_none() && !self.analyzer.pre_frames.is_empty() {
            self.spectral_ceiling = compute_spectral_ceiling(
                &self.analyzer.pre_frames, self.bar_count,
                self.min_freq, self.max_freq.min(self.analyzer.sample_rate as f32 / 2.0),
            );
        }

        // Current chord — only computed when the overlay is actually visible.
        if self.show_chord {
            if let Some(ref analysis) = self.track_analysis {
                if !analysis.chord_timeline.is_empty() && analysis.chord_step_secs > 0.0 {
                    let step = (elapsed_secs / analysis.chord_step_secs as f64) as usize;
                    let step = step.min(analysis.chord_timeline.len().saturating_sub(1));
                    self.current_chord = chord_name(analysis.chord_timeline[step]).to_string();
                }
            } else if self.mode == SpectrumMode::RealTime && !self.analyzer.last_fft_norms.is_empty() {
                // Live path: EMA-smooth the chroma before matching to reduce jitter.
                let fc = compute_chroma(
                    &self.analyzer.last_fft_norms,
                    self.analyzer.sample_rate,
                    self.analyzer.fft_size * self.analyzer.pad_factor,
                );
                const ALPHA: f32 = 0.8;
                for (e, &c) in self.chroma_ema.iter_mut().zip(&fc) {
                    *e = *e * ALPHA + c * (1.0 - ALPHA);
                }
                self.current_chord = chord_name(match_chord(&self.chroma_ema)).to_string();
            }
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

        // Measure actual FFT rate (not UI repaint rate)
        if elapsed_since_last < 5.0 {
            let dt = elapsed_since_last as f32;
            self.current_fps = self.current_fps * 0.85 + (1.0 / dt) * 0.15;
        }
        self.last_fft_time = Some(now);

        // Only maintain the waterfall ring-buffer when the waterfall view is actually displayed.
        // Skipping it when not needed saves ~4KB/tick of allocation+shift work.
        self.analyzer.waterfall_enabled = self.style == VizStyle::Waterfall;

        match self.mode {
            SpectrumMode::RealTime => {
                self.analyzer.process_realtime();
                // Feed spectrogram and octave bands from fresh FFT norms.
                // Real-time uses 2× padding (not pad_factor) — match here.
                if !self.analyzer.last_fft_norms.is_empty() {
                    let eff_fft = self.analyzer.fft_size * 2;
                    self.spectrogram.push_frame(
                        &self.analyzer.last_fft_norms,
                        self.analyzer.sample_rate,
                        eff_fft,
                        self.min_freq, self.max_freq,
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
                    let alpha = self.smoothing;
                    for (i, &(fc, v)) in raw.iter().enumerate() {
                        self.octave_smoothed[i] = self.octave_smoothed[i] * alpha + v * (1.0 - alpha);
                        self.octave_bands[i] = (fc, self.octave_smoothed[i]);
                    }
                }
            }
            SpectrumMode::PreProcess => self.analyzer.tick_pre(elapsed_secs),
        }

        // Momentary LUFS (400 ms window from live sample buffer).
        // Throttled: run at most every other tick so at 30 fps it updates at ~15 fps —
        // plenty for a meter display. LUFS is also used by the main-window readout
        // so we keep it running even when the spectrum panel is not in phasescope mode.
        self.lufs_tick = self.lufs_tick.wrapping_add(1);
        if self.lufs_tick.is_multiple_of(2) {
            let window = (self.analyzer.sample_rate as f64 * 0.4) as usize;
            let lufs_slice: Vec<f32> = {
                let guard = self.sample_buf.lock().unwrap_or_else(|p| p.into_inner());
                if guard.len() >= window {
                    let start = guard.len() - window;
                    guard[start..].to_vec()
                } else {
                    Vec::new()
                }
            }; // lock released here
            if !lufs_slice.is_empty() {
                let mut kw = KWeightFilter::new(self.analyzer.sample_rate);
                let mean_sq: f64 = lufs_slice.iter()
                    .map(|&s| { let y = kw.process(s) as f64; y * y })
                    .sum::<f64>() / lufs_slice.len() as f64;
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
        // Reset smoothing so we don't smear from the old position
        let n = self.analyzer.bar_count;
        self.analyzer.magnitudes = vec![0.0; n];
        self.analyzer.smoothed = vec![0.0; n];
        self.analyzer.waterfall.clear();
        self.analyzer.waterfall_dirty = false;
        self.waterfall_texture = None;
        // In pre-process mode, immediately snap the display to the new position
        if self.mode == SpectrumMode::PreProcess && !self.analyzer.pre_frames.is_empty() {
            let frame = ((elapsed_secs * self.analyzer.pre_frame_rate) as usize)
                .min(self.analyzer.pre_frames.len().saturating_sub(1));
            self.analyzer.magnitudes = self.analyzer.pre_frames[frame].clone();
            self.analyzer.smoothed = self.analyzer.magnitudes.clone();
        }
        // Reset peak hold state so peaks don't hang from the old position
        self.reset_peaks();
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

        let mags = &self.analyzer.smoothed;
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
                // ── Row 1: mode + view + loudness ─────────────────────────
                ui.horizontal(|ui| {
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
                    ui.label("Loudness:");
                    let prev_loudness = self.loudness_mode.clone();
                    ui.selectable_value(&mut self.loudness_mode, LoudnessMode::Flat, "Flat");
                    ui.selectable_value(&mut self.loudness_mode, LoudnessMode::EqualLoudness, "ISO 226");
                    if self.loudness_mode != prev_loudness {
                        self.sync_params();
                    }
                    ui.separator();
                    ui.label("Chord:");
                    let chord_label = if self.show_chord {
                        egui::RichText::new("On").color(Color32::from_rgb(120, 220, 160))
                    } else {
                        egui::RichText::new("Off").color(Color32::from_gray(120))
                    };
                    if ui.selectable_label(self.show_chord, chord_label).clicked() {
                        self.show_chord = !self.show_chord;
                    }
                    ui.separator();
                    let eq_label = egui::RichText::new("🎛 EQ")
                        .color(if self.show_eq { Color32::from_rgb(100, 200, 255) } else { Color32::from_gray(120) });
                    if ui.selectable_label(self.show_eq, eq_label)
                        .on_hover_text("Toggle parametric EQ panel.\nClick on spectrum to add bands.\nDrag nodes to adjust.\nRight-click node to remove.")
                        .clicked()
                    {
                        self.show_eq = !self.show_eq;
                    }
                    ui.separator();
                    let dbg_label = egui::RichText::new("🐛 Debug")
                        .color(if self.show_debug { Color32::from_rgb(255, 180, 60) } else { Color32::from_gray(120) });
                    if ui.selectable_label(self.show_debug, dbg_label)
                        .on_hover_text("Toggle debug overlay (F3)")
                        .clicked()
                    {
                        self.show_debug = !self.show_debug;
                    }
                });

                // ── Row 2: bar count ──────────────────────────────────────
                ui.horizontal(|ui| {
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
                });

                // ── Row 3: FFT params (collapsible) ───────────────────────
                egui::CollapsingHeader::new("⚙ FFT Settings")
                    .default_open(false)
                    .show(ui, |ui| {
                        let mut rebuild = false;
                        ui.horizontal(|ui| {
                            ui.label("FFT size:");
                            for &sz in &[1024usize, 2048, 4096, 8192, 16384] {
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
                                        &self.bar_mapping, &self.interp_mode,
                                    ))
                                }).unwrap_or(false);
                                let label = if has_cache {
                                    egui::RichText::new(&text).color(Color32::from_rgb(100, 210, 100))
                                } else if is_auto {
                                    egui::RichText::new(&text).color(Color32::from_rgb(220, 180, 60))
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
                        {
                            let wf_green = |wf: WindowFn| -> egui::RichText {
                                let has = self.current_path.as_ref().map(|p| {
                                    self.cache_file_set.contains(&cache_path_for(
                                        p, self.bar_count, self.fft_size, self.pad_factor,
                                        self.overlap, &wf, self.min_freq, self.max_freq,
                                        &self.bar_mapping, &self.interp_mode,
                                    ))
                                }).unwrap_or(false);
                                let name = match wf {
                                    WindowFn::Hann    => "Hann",
                                    WindowFn::Hamming => "Hamming",
                                    WindowFn::Blackman => "Blackman",
                                    WindowFn::FlatTop  => "Flat-top",
                                };
                                if has { egui::RichText::new(name).color(Color32::from_rgb(100, 210, 100)) }
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
                                        &self.bar_mapping, &im,
                                    ))
                                }).unwrap_or(false);
                                let name = match im {
                                    InterpolationMode::None      => "None",
                                    InterpolationMode::Linear    => "Linear",
                                    InterpolationMode::CatmullRom => "Catmull-Rom",
                                    InterpolationMode::Pchip     => "PCHIP",
                                    InterpolationMode::Akima     => "Akima",
                                    InterpolationMode::Lanczos   => "Lanczos",
                                };
                                if has { egui::RichText::new(name).color(Color32::from_rgb(100, 210, 100)) }
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
                                        &self.bar_mapping, &self.interp_mode,
                                    );
                                    self.cache_file_set.contains(&candidate)
                                }).unwrap_or(false);
                                let rich = if has_cache {
                                    egui::RichText::new(label_text).color(Color32::from_rgb(100, 210, 100))
                                } else {
                                    egui::RichText::new(label_text)
                                };
                                let btn = ui.selectable_label(self.pad_factor == pf, rich);
                                let btn = match pf {
                                    1  => btn.on_hover_text("No padding — rely entirely on interpolation."),
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
                                        &self.bar_mapping, &self.interp_mode,
                                    ))
                                }).unwrap_or(false);
                                let rich = if has_cache {
                                    egui::RichText::new(text).color(Color32::from_rgb(100, 210, 100))
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
                                        &bm, &self.interp_mode,
                                    ))
                                }).unwrap_or(false);
                                let name = match bm {
                                    BarMappingMode::FlatOverlap => "Flat",
                                    BarMappingMode::Gaussian    => "Gaussian",
                                    BarMappingMode::Cqt         => "CQT",
                                };
                                if has { egui::RichText::new(name).color(Color32::from_rgb(100, 210, 100)) }
                                else   { egui::RichText::new(name) }
                            };
                            let lbl_flat  = bm_green(BarMappingMode::FlatOverlap);
                            let lbl_gauss = bm_green(BarMappingMode::Gaussian);
                            let lbl_cqt   = bm_green(BarMappingMode::Cqt);
                            ui.horizontal(|ui| {
                                ui.label("Bar mapping:");
                                let prev = self.bar_mapping.clone();
                                ui.selectable_value(&mut self.bar_mapping, BarMappingMode::FlatOverlap, lbl_flat)
                                    .on_hover_text("Equal weight to all FFT bins within the bar's frequency range.");
                                ui.selectable_value(&mut self.bar_mapping, BarMappingMode::Gaussian, lbl_gauss)
                                    .on_hover_text("Bins near the bar's centre frequency weighted more heavily. More natural.");
                                ui.selectable_value(&mut self.bar_mapping, BarMappingMode::Cqt, lbl_cqt)
                                    .on_hover_text("Constant-Q Transform — Hann kernel with bandwidth ∝ frequency.\nEach bar has identical relative frequency resolution. Best for music.");
                                if self.bar_mapping != prev {
                                    self.analyzer.bar_mapping = self.bar_mapping.clone();
                                    self.try_load_or_flag_reanalysis();
                                }
                            });
                        }
                        ui.horizontal(|ui| {
                            ui.label("Smoothing:");
                            ui.add(egui::Slider::new(&mut self.smoothing, 0.0..=0.97)
                                .step_by(0.01));
                            self.analyzer.smoothing = self.smoothing;
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
                            ui.add(egui::Slider::new(&mut self.max_fps, 1.0..=120.0)
                                .step_by(1.0).suffix(" fps"));
                        });
                        if rebuild {
                            self.analyzer.rebuild_fft();
                        }
                    });

                // ── Row 4: pre-process cache controls ────────────────────
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
                                .size(11.0).color(Color32::from_rgb(255, 200, 60)));
                            let has_path = self.current_path.is_some();
                            if ui.add_enabled(has_path && !analyzing,
                                egui::Button::new("🔄 Re-analyze now")).clicked()
                                && let Some(ref p) = self.current_path.clone() {
                                let cache = cache_path_for(p, self.bar_count, self.fft_size, self.pad_factor, self.overlap, &self.window_fn, self.min_freq, self.max_freq, &self.bar_mapping, &self.interp_mode);
                                let _ = std::fs::remove_file(&cache);
                                self.analyzer.pre_frames.clear();
                                self.analyzer.start_preprocess(p.clone());
                                self.needs_reanalysis = false;
                                self.status_msg = String::new();
                            }
                        });
                    }
                    ui.horizontal(|ui| {
                        let analyzing = self.analyzer.is_analyzing.load(Ordering::Relaxed);
                        let has_path  = self.current_path.is_some();
                        if ui.add_enabled(has_path && !analyzing,
                            egui::Button::new("🗑 Clear Cache")).clicked()
                            && let Some(ref p) = self.current_path.clone() {
                            let cache = cache_path_for(p, self.bar_count, self.fft_size, self.pad_factor, self.overlap, &self.window_fn, self.min_freq, self.max_freq, &self.bar_mapping, &self.interp_mode);
                            let existed = cache.exists();
                            let _ = std::fs::remove_file(&cache);
                            self.analyzer.pre_frames.clear();
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
                            let cache = cache_path_for(p, self.bar_count, self.fft_size, self.pad_factor, self.overlap, &self.window_fn, self.min_freq, self.max_freq, &self.bar_mapping, &self.interp_mode);
                            let _ = std::fs::remove_file(&cache);
                            self.analyzer.pre_frames.clear();
                            self.analyzer.start_preprocess(p.clone());
                            self.needs_reanalysis = false;
                            self.status_msg = String::new();
                        }
                        if analyzing {
                            let pct = self.analyzer.analysis_progress.load(Ordering::Relaxed);
                            ui.spinner();
                            ui.label(egui::RichText::new(format!("Analyzing… {}%", pct))
                                .size(11.0).color(Color32::from_gray(160)));
                        } else if !self.status_msg.is_empty() {
                            ui.label(egui::RichText::new(&self.status_msg)
                                .size(11.0).color(Color32::from_gray(160)));
                        }
                    });
                    // Cache size — refreshed at most once per 2 s
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
                            .size(10.0).color(Color32::from_gray(110)));
                        let analyzing = self.analyzer.is_analyzing.load(Ordering::Relaxed);
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
                            self.analyzer.pre_frames.clear();
                            self.needs_reanalysis = self.mode == SpectrumMode::PreProcess
                                && self.current_path.is_some();
                            self.cache_stats_at = None;
                            self.status_msg = "All caches cleared.".into();
                        }
                    });
                }

                // Bar gap slider (only meaningful for Bars style)
                if self.style == VizStyle::Bars {
                    ui.horizontal(|ui| {
                        ui.label("Bar gap:");
                        ui.add(egui::Slider::new(&mut self.bar_gap, 0.0..=12.0)
                            .step_by(1.0).suffix(" px"))
                            .on_hover_text(
                                "Physical-pixel gap between bars.\n\
                                 0 = no gap (solid fill). Higher values give a more separated look."
                            );
                    });
                }

                // ── Peak Hold settings (Bars only) ────────────────────────
                if self.style == VizStyle::Bars {
                    egui::CollapsingHeader::new("📌 Peak Hold")
                        .default_open(false)
                        .show(ui, |ui| {
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
                egui::CollapsingHeader::new("🖼 Album Art")
                    .default_open(false)
                    .show(ui, |ui| {
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
                                .color(Color32::from_gray(160)));
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
                                .size(10.0).color(Color32::from_gray(100)));
                        }
                    });

                // ── EQ panel ──────────────────────────────────────────────
                if self.show_eq {
                    egui::CollapsingHeader::new("🎛 Parametric EQ")
                        .default_open(true)
                        .show(ui, |ui| {
                            // Global controls row
                            ui.horizontal(|ui| {
                                let eq_on = { self.eq_state.lock().unwrap().enabled };
                                let on_label = egui::RichText::new(if eq_on { "ON" } else { "OFF" })
                                    .color(if eq_on { Color32::from_rgb(100, 220, 120) } else { Color32::from_gray(120) });
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
                                ui.separator();
                                // Bake to cache
                                let has_path = self.current_path.is_some();
                                let analyzing = self.analyzer.is_analyzing.load(Ordering::Relaxed);
                                if ui.add_enabled(has_path && !analyzing,
                                    egui::Button::new("💾 Bake to Cache"))
                                    .on_hover_text("Re-analyze with EQ applied and cache result.\nCreates a separate cache file keyed to these EQ settings.")
                                    .clicked()
                                    && let Some(ref p) = self.current_path.clone() {
                                    let eq = self.eq_state.lock().unwrap();
                                    let fp = eq.fingerprint();
                                    drop(eq);
                                    let mut cache = cache_path_for(p, self.bar_count, self.fft_size, self.pad_factor, self.overlap, &self.window_fn, self.min_freq, self.max_freq, &self.bar_mapping, &self.interp_mode);
                                    // Append EQ fingerprint to filename stem
                                    let stem = cache.file_stem().unwrap_or_default().to_string_lossy().to_string();
                                    cache.set_file_name(format!("{}_eq{:016x}.spectrumcache", stem, fp));
                                    let _ = std::fs::remove_file(&cache);
                                    self.analyzer.pre_frames.clear();
                                    self.analyzer.start_preprocess(p.clone());
                                    self.needs_reanalysis = false;
                                }
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
                                        ui.label(egui::RichText::new("⚠ Unsaved changes — ").color(Color32::from_rgb(255, 200, 80)));
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
                                    ui.label(egui::RichText::new("Preset:").size(11.0).color(Color32::from_gray(160)));

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
                                    ui.label(egui::RichText::new("#").size(10.0).color(Color32::from_gray(130)));
                                    ui.label(egui::RichText::new("Type").size(10.0).color(Color32::from_gray(130)));
                                    ui.label(egui::RichText::new("Freq (Hz)").size(10.0).color(Color32::from_gray(130)));
                                    ui.label(egui::RichText::new("Gain (dB)").size(10.0).color(Color32::from_gray(130)));
                                    ui.label(egui::RichText::new("Q").size(10.0).color(Color32::from_gray(130)));
                                    ui.label(egui::RichText::new("On").size(10.0).color(Color32::from_gray(130)));
                                    ui.label(egui::RichText::new("Del").size(10.0).color(Color32::from_gray(130)));
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
                                    .size(10.0).color(Color32::from_gray(110)));
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
                    let eq_snap = if self.show_eq {
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

                    match self.style {
                        VizStyle::Bars => {
                            if has_art && matches!(art_cfg.spectrum_mode, ArtSpectrumMode::Mask) {
                                if let Some((tex_id, art_w, art_h)) = self.current_art {
                                    draw_bars_art_mask(
                                        &painter, mags, plot_rect, self.bar_gap,
                                        tex_id, art_w, art_h, &art_cfg.fit,
                                        &art_cfg.mask_mode, art_cfg.mask_brightness,
                                    );
                                } else {
                                    draw_bars(&painter, mags, plot_rect, self.bar_gap);
                                }
                            } else {
                                draw_bars(&painter, mags, plot_rect, self.bar_gap);
                            }
                            if self.peak_config.enabled && !self.peak_vals.is_empty() {
                                draw_peak_hold(
                                    &painter, &self.peak_vals, &self.peak_alphas,
                                    plot_rect, self.bar_gap, &self.peak_config,
                                );
                            }
                        }
                        VizStyle::Line       => draw_line(&painter, mags, plot_rect,
                                                 Color32::from_rgb(80, 200, 255)),
                        VizStyle::FilledArea => draw_filled(&painter, mags, plot_rect),
                        VizStyle::Waterfall  => {
                            if self.analyzer.waterfall_dirty {
                                if let Some(img) = waterfall_to_color_image(&self.analyzer.waterfall) {
                                    if let Some(ref mut th) = self.waterfall_texture {
                                        th.set(img, egui::TextureOptions::NEAREST);
                                    } else {
                                        let th = vp_ctx.load_texture(
                                            "moosik_waterfall", img,
                                            egui::TextureOptions::NEAREST,
                                        );
                                        self.waterfall_texture = Some(th);
                                    }
                                }
                                self.analyzer.waterfall_dirty = false;
                            }
                            if let Some(ref th) = self.waterfall_texture {
                                painter.image(
                                    th.id(), plot_rect,
                                    egui::Rect::from_min_max(
                                        egui::Pos2::ZERO,
                                        egui::Pos2::new(1.0, 1.0),
                                    ),
                                    Color32::WHITE,
                                );
                            }
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
                                self.analyzer.sample_rate, self.min_freq, self.max_freq,
                            );
                        }
                        VizStyle::Phasescope => {
                            // Use the snapshot captured in tick() — no lock or clone here
                            draw_phasescope(&painter, &self.phasescope_frames, plot_rect, self.correlation);
                        }
                    }
                    // Axis labels — skip for spectrogram (its own freq axis is baked in),
                    // octave bands (draws its own labels below bars), and phasescope.
                    if !matches!(self.style, VizStyle::Spectrogram | VizStyle::OctaveBands | VizStyle::Phasescope) {
                        draw_db_labels(&painter, plot_rect);
                    }
                    if !matches!(self.style, VizStyle::Spectrogram | VizStyle::Phasescope) {
                        draw_freq_labels(&painter, plot_rect, self.analyzer.sample_rate, self.min_freq, self.max_freq);
                    }
                    // ── EQ overlay and node interaction ──────────────────────
                    if self.show_eq && !matches!(self.style, VizStyle::Phasescope | VizStyle::Waterfall | VizStyle::Spectrogram) {
                        let (bands, sr) = {
                            let eq = self.eq_state.lock().unwrap();
                            (eq.bands.clone(), eq.sample_rate)
                        };
                        let draw_curve = matches!(self.eq_overlay, EqOverlayMode::Curve | EqOverlayMode::Both);
                        draw_eq_overlay(&painter, &bands, sr, plot_rect,
                            self.min_freq, self.max_freq, draw_curve,
                            self.eq_hovered_node, self.eq_dragging_node);

                        // Mouse hit-test: find nearest node within 14px
                        let log_min  = self.min_freq.log10();
                        let log_span = (self.max_freq.log10() - log_min).max(1e-6);

                        if let Some(ptr) = spec_response.hover_pos() {
                            self.eq_hovered_node = None;
                            for (i, band) in bands.iter().enumerate() {
                                if !band.enabled { continue; }
                                let np = eq_node_pos(band, plot_rect, self.min_freq, self.max_freq);
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
                                    let freq_delta = delta.x / plot_rect.width() * log_span;
                                    band.freq = (band.freq * 10f32.powf(freq_delta)).clamp(20.0, 20_000.0);
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
                            let freq = 10f32.powf(log_min + t * log_span).clamp(20.0, 20_000.0);
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

                    // FPS overlay — top-right corner of viz area
                    let fps_text = format!("{:.0} fps", self.current_fps);
                    painter.text(
                        Pos2::new(rect.right() - 6.0, rect.top() + 4.0),
                        egui::Align2::RIGHT_TOP,
                        fps_text,
                        egui::FontId::monospace(10.0),
                        Color32::from_rgba_unmultiplied(180, 180, 180, 120),
                    );
                    if self.style != VizStyle::Phasescope {
                        let lufs_str = if self.momentary_lufs.is_finite() {
                            format!("{:.1} LUFS", self.momentary_lufs)
                        } else {
                            "— LUFS".to_string()
                        };
                        painter.text(
                            Pos2::new(rect.left() + 6.0, rect.top() + 4.0),
                            egui::Align2::LEFT_TOP,
                            lufs_str,
                            egui::FontId::monospace(10.0),
                            Color32::from_rgba_unmultiplied(180, 220, 180, 160),
                        );
                    }
                    // Debug overlay (F3)
                    if self.show_debug {
                        let interp_name = format!("{:?}", self.interp_mode);
                        let mapping_name = format!("{:?}", self.bar_mapping);
                        let mode_name = format!("{:?}", self.mode);
                        let analyzing = self.analyzer.is_analyzing.load(Ordering::Relaxed);
                        let pct = self.analyzer.analysis_progress.load(Ordering::Relaxed);
                        let lines = vec![
                            format!("── Spectrum Debug ──────────────────"),
                            format!("Mode:        {}", mode_name),
                            format!("FFT size:    {}  pad: {}×  padded: {}", self.fft_size, self.pad_factor, self.fft_size * self.pad_factor),
                            format!("Overlap:     {:.0}%  hop: {} smp", self.overlap * 100.0,
                                ((self.fft_size as f32 * (1.0 - self.overlap)).round() as usize).max(1)),
                            format!("Interp:      {}  mapping: {}", interp_name, mapping_name),
                            format!("Bars:        {}  sr: {} Hz", self.bar_count, self.analyzer.sample_rate),
                            format!("freq range:  {:.0}–{:.0} Hz", self.min_freq, self.max_freq),
                            format!("── Pre-process ─────────────────────"),
                            format!("Analyzing:   {}  progress: {}%", analyzing, pct),
                            format!("Frames:      {}  frame_rate: {:.2} fps", self.analyzer.pre_frames.len(), self.analyzer.pre_frame_rate),
                            format!("── Runtime ─────────────────────────"),
                            format!("FPS:         {:.1}", self.current_fps),
                            format!("LUFS:        {:.2}", self.momentary_lufs),
                            format!("Correlation: {:.3}", self.correlation),
                            format!("FFT norms:   {} bins", self.analyzer.last_fft_norms.len()),
                            format!("Chord:       {} (show={})", self.current_chord, self.show_chord),
                        ];
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

                    // Chord overlay
                    if self.show_chord && !self.current_chord.is_empty() {
                        let cx = rect.center().x;
                        let cy = rect.bottom() - 20.0;
                        // Shadow
                        painter.text(
                            Pos2::new(cx + 1.0, cy + 1.0),
                            egui::Align2::CENTER_CENTER,
                            &self.current_chord,
                            egui::FontId::proportional(38.0),
                            Color32::from_rgba_unmultiplied(0, 0, 0, 180),
                        );
                        painter.text(
                            Pos2::new(cx, cy),
                            egui::Align2::CENTER_CENTER,
                            &self.current_chord,
                            egui::FontId::proportional(38.0),
                            Color32::from_rgba_unmultiplied(255, 240, 120, 230),
                        );
                    }
                }
            });
        });
    }
}

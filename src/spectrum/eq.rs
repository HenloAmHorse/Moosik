use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Parametric EQ — types, biquad math, shared state
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum BandKind { Peaking, LowShelf, HighShelf, HighPass, LowPass, Notch }

impl BandKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Peaking   => "Peak",
            Self::LowShelf  => "Low S",
            Self::HighShelf => "High S",
            Self::HighPass  => "HP",
            Self::LowPass   => "LP",
            Self::Notch     => "Notch",
        }
    }
    pub fn has_gain(&self) -> bool {
        matches!(self, Self::Peaking | Self::LowShelf | Self::HighShelf)
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct EqBand {
    pub freq:    f32,
    pub gain_db: f32,   // ignored for HP / LP / Notch
    pub q:       f32,
    pub kind:    BandKind,
    pub enabled: bool,
}

impl EqBand {
    pub fn default_peak(freq: f32) -> Self {
        Self { freq, gain_db: 0.0, q: 1.0, kind: BandKind::Peaking, enabled: true }
    }
}

/// How the EQ response is overlaid on the spectrum display.
#[derive(Clone, PartialEq)]
pub enum EqOverlayMode {
    /// Draw the combined EQ curve as a coloured line.
    Curve,
    /// Multiply bar heights by EQ gain — bars show what you hear.
    ApplyToBars,
    /// Both simultaneously.
    Both,
}

/// Shared EQ state between the UI thread and the audio thread.
pub struct EqState {
    pub bands:       Vec<EqBand>,
    /// Gain before the bands, dB (an AutoEq profile's "Preamp"), so boosts
    /// do not clip.
    pub preamp_db:   f32,
    /// Headphone crossfeed: (cut-off Hz, feed dB), or off.
    pub crossfeed:   Option<(f32, f32)>,
    /// Room correction: an impulse response convolved after the bands.
    pub fir:         Option<Arc<super::fir::Fir>>,
    pub enabled:     bool,
    pub sample_rate: u32,
    /// Incremented on every change so EqSource can detect updates cheaply.
    pub version:     u64,
}

impl EqState {
    pub fn new() -> Self {
        Self { bands: Vec::new(), preamp_db: 0.0, crossfeed: None, fir: None, enabled: true, sample_rate: 44_100, version: 0 }
    }
    /// Call after mutating bands or sample_rate.
    pub fn bump(&mut self) { self.version = self.version.wrapping_add(1); }
    /// True when EQ is enabled and at least one band is active.
    pub fn is_active(&self) -> bool {
        // Crossfeed and room correction are not the EQ: its switch leaves them on.
        (self.enabled && (self.bands.iter().any(|b| b.enabled) || self.preamp_db != 0.0)) || self.crossfeed.is_some() || self.fir.is_some()
    }
}

pub type EqStateHandle = Arc<Mutex<EqState>>;

// ---------------------------------------------------------------------------
// EQ Preset system
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize)]
pub struct EqPreset {
    pub id:    u64,
    pub name:  String,
    pub bands: Vec<EqBand>,
    #[serde(default)] pub preamp_db: f32,
}

/// Crossfeed strengths: name, cut-off (Hz) and feed (dB), after bs2b's
/// presets.
pub const CROSSFEED: [(&str, f32, f32); 3] = [("Light", 700.0, 4.5), ("Chu Moy", 700.0, 6.0), ("Jan Meier", 650.0, 9.5)];

/// Headphone crossfeed after Boris Mikhaylov's bs2b: each ear also hears
/// the other channel, low-passed and quieter, and its own a little
/// high-boosted, as from speakers in front. Stereo only.
pub struct Crossfeed {
    a0_lo: f64,
    b1_lo: f64,
    a0_hi: f64,
    a1_hi: f64,
    b1_hi: f64,
    lo: [f64; 2],
    hi: [f64; 2],
    prev: [f64; 2],
}

impl Crossfeed {
    pub fn new(rate: u32, fcut: f32, feed_db: f32) -> Self {
        let fs = rate as f64;
        let feed = feed_db as f64;
        let gb_lo = feed * -5.0 / 6.0 - 3.0;
        let gb_hi = feed / 6.0 - 3.0;
        let g_lo = 10f64.powf(gb_lo / 20.0);
        let g_hi = 1.0 - 10f64.powf(gb_hi / 20.0);
        let fc_lo = fcut as f64;
        let fc_hi = fc_lo * 2f64.powf((gb_lo - 20.0 * g_hi.log10()) / 12.0);
        let x = (-2.0 * std::f64::consts::PI * fc_lo / fs).exp();
        let (b1_lo, a0_lo) = (x, g_lo * (1.0 - x));
        let x = (-2.0 * std::f64::consts::PI * fc_hi / fs).exp();
        let (b1_hi, a0_hi, a1_hi) = (x, 1.0 - g_hi * (1.0 - x), -x);
        Crossfeed { a0_lo, b1_lo, a0_hi, a1_hi, b1_hi, lo: [0.0; 2], hi: [0.0; 2], prev: [0.0; 2] }
    }

    /// One stereo frame.
    pub fn frame(&mut self, l: f64, r: f64) -> (f64, f64) {
        let x = [l, r];
        for c in 0..2 {
            self.lo[c] = self.a0_lo * x[c] + self.b1_lo * self.lo[c];
            self.hi[c] = self.a0_hi * x[c] + self.a1_hi * self.prev[c] + self.b1_hi * self.hi[c];
        }
        self.prev = x;
        (self.hi[0] + self.lo[1], self.hi[1] + self.lo[0])
    }
}

/// The most bands the EQ holds.
pub const MAX_BANDS: usize = 32;

/// An AutoEq / Equalizer APO parametric profile ("ParametricEQ.txt"):
/// its preamp and its filters as bands. Lines it does not know are skipped;
/// a filter marked OFF is kept, switched off.
pub fn parse_autoeq(text: &str) -> Result<(f32, Vec<EqBand>), String> {
    let mut preamp = 0.0;
    let mut bands = Vec::new();
    let num = |w: &[&str], key: &str| -> Option<f32> {
        w.iter().position(|t| t.eq_ignore_ascii_case(key)).and_then(|i| w.get(i + 1)).and_then(|v| v.parse().ok())
    };
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Preamp:") {
            preamp = rest.split_whitespace().next().and_then(|v| v.parse().ok()).ok_or("an unreadable Preamp line")?;
            continue;
        }
        let Some(rest) = line.strip_prefix("Filter") else { continue };
        let Some((_, spec)) = rest.split_once(':') else { continue };
        let w: Vec<&str> = spec.split_whitespace().collect();
        if w.len() < 2 {
            continue;
        }
        let enabled = w[0].eq_ignore_ascii_case("ON");
        let kind = match w[1].to_ascii_uppercase().as_str() {
            "PK" | "PEQ" => BandKind::Peaking,
            "LS" | "LSC" | "LSQ" => BandKind::LowShelf,
            "HS" | "HSC" | "HSQ" => BandKind::HighShelf,
            "HP" | "HPQ" => BandKind::HighPass,
            "LP" | "LPQ" => BandKind::LowPass,
            "NO" => BandKind::Notch,
            other => return Err(format!("a filter type Moosik does not have: {other}")),
        };
        let freq = num(&w, "Fc").ok_or("a filter without Fc")?;
        let gain_db = num(&w, "Gain").unwrap_or(0.0);
        let mut q = num(&w, "Q").unwrap_or(if matches!(kind, BandKind::Peaking | BandKind::Notch) { 1.0 } else { std::f32::consts::FRAC_1_SQRT_2 });
        if !(freq > 0.0 && freq.is_finite() && q > 0.0 && q.is_finite() && gain_db.is_finite()) {
            return Err("a filter with an impossible value".into());
        }
        if q < 0.01 {
            return Err(format!("a filter with Q {q}; Moosik's EQ goes down to Q 0.01"));
        }
        if matches!(kind, BandKind::LowShelf | BandKind::HighShelf) {
            q = shelf_slope(q as f64, gain_db);
        }
        bands.push(EqBand { freq, gain_db, q, kind, enabled });
    }
    if bands.is_empty() {
        return Err("no filters in it: not a parametric EQ profile".into());
    }
    if bands.len() > MAX_BANDS {
        return Err(format!("{} filters; Moosik's EQ holds {MAX_BANDS}", bands.len()));
    }
    Ok((preamp, bands))
}

/// Whether a preset lives in the global list or is track-specific.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum PresetScope { Global, Local }

/// A lightweight reference to a preset — stored in last_used.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetRef { pub scope: PresetScope, pub id: u64 }

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct EqPresetLibrary {
    #[serde(default)] pub global:     Vec<EqPreset>,
    #[serde(default)] pub per_track:  HashMap<String, Vec<EqPreset>>,
    pub default_id: Option<u64>,
    #[serde(default)] pub last_used:  HashMap<String, PresetRef>,
    #[serde(default)] pub next_id:    u64,
}

impl EqPresetLibrary {
    fn preset_path() -> PathBuf {
        home_dir().join(".moosik").join("eq_presets.json")
    }

    pub fn load() -> Self {
        // A test build never reads the owner's presets. With a default preset
        // set there, `on_play` applied it, and the next track change wrote the
        // library back to the real profile from inside the test suite.
        if cfg!(test) {
            return Self::default();
        }
        std::fs::read_to_string(Self::preset_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        // Nor writes them, whatever a test does to the library in memory.
        if cfg!(test) {
            return;
        }
        let path = Self::preset_path();
        if let Some(p) = path.parent() { let _ = std::fs::create_dir_all(p); }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, json);
        }
    }

    pub fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn local_presets(&self, track: &str) -> &[EqPreset] {
        self.per_track.get(track).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn find(&self, r: &PresetRef, track: Option<&str>) -> Option<&EqPreset> {
        match r.scope {
            PresetScope::Global => self.global.iter().find(|p| p.id == r.id),
            PresetScope::Local  => self.per_track.get(track?)?.iter().find(|p| p.id == r.id),
        }
    }

    pub fn find_mut(&mut self, r: &PresetRef, track: Option<&str>) -> Option<&mut EqPreset> {
        match r.scope {
            PresetScope::Global => self.global.iter_mut().find(|p| p.id == r.id),
            PresetScope::Local  => self.per_track.get_mut(track?)?.iter_mut().find(|p| p.id == r.id),
        }
    }

    pub fn delete(&mut self, r: &PresetRef, track: Option<&str>) {
        match r.scope {
            PresetScope::Global => {
                self.global.retain(|p| p.id != r.id);
                if self.default_id == Some(r.id) { self.default_id = None; }
            }
            PresetScope::Local => {
                if let Some(k) = track && let Some(v) = self.per_track.get_mut(k) { v.retain(|p| p.id != r.id); }
            }
        }
        self.last_used.retain(|_, lu| lu != r);
    }

    pub fn auto_name(&self, scope: &PresetScope, track: Option<&str>) -> String {
        let n = match scope {
            PresetScope::Global => self.global.len(),
            PresetScope::Local  => track.and_then(|t| self.per_track.get(t)).map(|v| v.len()).unwrap_or(0),
        };
        format!("Preset {}", n + 1)
    }
}

/// Per-band colour used for nodes and individual curves.
pub fn eq_band_color(idx: usize) -> egui::Color32 {
    use egui::Color32;
    const PALETTE: &[Color32] = &[
        Color32::from_rgb(255, 120,  80),
        Color32::from_rgb(255, 200,  60),
        Color32::from_rgb(120, 255, 100),
        Color32::from_rgb( 60, 220, 255),
        Color32::from_rgb(160, 100, 255),
        Color32::from_rgb(255, 100, 200),
        Color32::from_rgb(255, 160,  60),
        Color32::from_rgb(100, 255, 200),
    ];
    PALETTE[idx % PALETTE.len()]
}

// ---------------------------------------------------------------------------
// Biquad math — Audio EQ Cookbook formulas (R. Bristow-Johnson)
// ---------------------------------------------------------------------------

/// Moosik's shelves take the cookbook's shelf slope S in `q`, as they always
/// have, so presets keep their sound. AutoEq and Equalizer APO give a
/// shelf's Q instead: this is the S that makes the same filter, coefficient
/// for coefficient, at that gain.
pub fn shelf_slope(q: f64, gain_db: f32) -> f32 {
    let a = 10f64.powf(gain_db as f64 / 40.0);
    (1.0 / (1.0 + (1.0 / (q * q) - 2.0) / (a + 1.0 / a))) as f32
}

/// Compute normalised biquad coefficients (b[3], a[2]) for one EQ band.
/// The existing `Biquad` struct takes exactly these arrays.
pub fn eq_biquad_coeffs(band: &EqBand, sr: u32) -> ([f64; 3], [f64; 2]) {
    use std::f64::consts::PI;
    let sr  = sr as f64;
    let f0  = (band.freq as f64).clamp(1.0, sr * 0.499);
    let q   = (band.q   as f64).max(0.01);
    let w0  = 2.0 * PI * f0 / sr;
    let cw  = w0.cos();
    let sw  = w0.sin();
    let al  = sw / (2.0 * q);

    let (b0, b1, b2, a0, a1, a2) = match band.kind {
        BandKind::Peaking => {
            let a = 10f64.powf(band.gain_db as f64 / 40.0);
            (1.0 + al*a, -2.0*cw, 1.0 - al*a,
             1.0 + al/a, -2.0*cw, 1.0 - al/a)
        }
        BandKind::LowShelf => {
            let a  = 10f64.powf(band.gain_db as f64 / 40.0);
            let as_ = (sw / 2.0) * shelf_root(a, band.q);
            ( a*((a+1.0)-(a-1.0)*cw+2.0*a.sqrt()*as_),
              2.0*a*((a-1.0)-(a+1.0)*cw),
              a*((a+1.0)-(a-1.0)*cw-2.0*a.sqrt()*as_),
              (a+1.0)+(a-1.0)*cw+2.0*a.sqrt()*as_,
              -2.0*((a-1.0)+(a+1.0)*cw),
              (a+1.0)+(a-1.0)*cw-2.0*a.sqrt()*as_ )
        }
        BandKind::HighShelf => {
            let a  = 10f64.powf(band.gain_db as f64 / 40.0);
            let as_ = (sw / 2.0) * shelf_root(a, band.q);
            ( a*((a+1.0)+(a-1.0)*cw+2.0*a.sqrt()*as_),
              -2.0*a*((a-1.0)+(a+1.0)*cw),
              a*((a+1.0)+(a-1.0)*cw-2.0*a.sqrt()*as_),
              (a+1.0)-(a-1.0)*cw+2.0*a.sqrt()*as_,
              2.0*((a-1.0)-(a+1.0)*cw),
              (a+1.0)-(a-1.0)*cw-2.0*a.sqrt()*as_ )
        }
        BandKind::HighPass => (
            (1.0+cw)/2.0, -(1.0+cw), (1.0+cw)/2.0,
            1.0+al, -2.0*cw, 1.0-al,
        ),
        BandKind::LowPass => (
            (1.0-cw)/2.0,  1.0-cw,  (1.0-cw)/2.0,
            1.0+al, -2.0*cw, 1.0-al,
        ),
        BandKind::Notch => (
            1.0, -2.0*cw, 1.0,
            1.0+al, -2.0*cw, 1.0-al,
        ),
    };
    ([b0/a0, b1/a0, b2/a0], [a1/a0, a2/a0])
}

/// A shelf's `sqrt((A + 1/A)(1/S - 1) + 2)`, which is 1/Q for the shelf's
/// Q. The slope is not floored at 0.01 as a peak's Q is: that took gentle
/// shelves given by their Q (Q 0.05 at +6 dB is a slope of 0.0053) for
/// steeper ones. This is bounded instead, at what that floor bounds a peak
/// to (1/Q at most 100), so every shelf of Q 0.01 or more is made as given.
/// From a slope of 0.01 up, the bound is never reached below 80 dB of gain.
fn shelf_root(a: f64, s: f32) -> f64 {
    ((a + 1.0/a) * (1.0/(s as f64).max(0.0) - 1.0) + 2.0).clamp(0.0, 1e4).sqrt()
}

/// Frequency response magnitude of a single normalised biquad in dB.
pub fn biquad_response_db(b: [f64; 3], a: [f64; 2], freq: f32, sr: u32) -> f32 {
    let w  = std::f64::consts::TAU * freq as f64 / sr as f64;
    let cw = w.cos();  let sw = w.sin();
    let c2 = (2.0*w).cos(); let s2 = (2.0*w).sin();
    let nr = b[0] + b[1]*cw + b[2]*c2;
    let ni = -(b[1]*sw + b[2]*s2);
    let dr = 1.0 + a[0]*cw + a[1]*c2;
    let di = -(a[0]*sw + a[1]*s2);
    let gain_sq = (nr*nr + ni*ni) / (dr*dr + di*di).max(1e-30);
    (10.0 * gain_sq.max(1e-30).log10()) as f32
}

/// True when two band lists are perceptually identical (within rounding tolerances).
pub fn bands_equal(a: &[EqBand], b: &[EqBand]) -> bool {
    if a.len() != b.len() { return false; }
    a.iter().zip(b).all(|(x, y)| {
        x.kind == y.kind && x.enabled == y.enabled
            && (x.freq    - y.freq   ).abs() < 0.5
            && (x.gain_db - y.gain_db).abs() < 0.05
            && (x.q       - y.q      ).abs() < 0.005
    })
}

// ---------------------------------------------------------------------------
// EqSource — rodio Source wrapper that applies the biquad chain in real time
// ---------------------------------------------------------------------------

pub struct EqSource<S: rodio::Source<Item = f32>> {
    inner:         S,
    eq:            EqStateHandle,
    // Per-band, per-channel stateful biquad filters.
    // Rebuilt whenever EqState::version changes.
    filters_l:     Vec<Biquad>,
    filters_r:     Vec<Biquad>,
    local_version: u64,
    local_enabled: bool,
    channels:      u16,
    /// The preamp as a factor.
    gain:          f64,
    crossfeed:     Option<Crossfeed>,
    local_crossfeed: Option<(f32, f32)>,
    conv:          Option<super::fir::Convolver>,
    /// Which response `conv` was built from (by address).
    local_fir:     usize,
    /// With crossfeed, a frame's right sample waits here for its turn.
    held:          Option<f32>,
    ch_idx:        u16,   // 0 = left, 1 = right (for stereo interleaving)
    // Check for EQ updates every N samples to avoid per-sample mutex contention.
    check_counter: u16,
}

impl<S> EqSource<S>
where S: rodio::Source<Item = f32>
{
    pub fn new(inner: S, eq: EqStateHandle) -> Self {
        let channels = inner.channels();
        let sr = inner.sample_rate();
        let mut this = Self {
            inner, eq,
            filters_l: Vec::new(), filters_r: Vec::new(),
            local_version: u64::MAX,   // force resync on first sample
            local_enabled: false,
            channels,
            gain: 1.0,
            crossfeed: None,
            local_crossfeed: None,
            conv: None,
            local_fir: 0,
            held: None,
            ch_idx: 0,
            check_counter: 0,
        };
        // Initial sync — blocking is fine here, audio thread hasn't started yet.
        {
            let mut state = this.eq.lock().unwrap();
            state.sample_rate = sr;
            state.bump();
            // Build filters inline so we never block in next().
            this.local_version = state.version;
            this.local_enabled = state.enabled;
            this.gain = 10f64.powf(state.preamp_db as f64 / 20.0);
            this.local_crossfeed = state.crossfeed.filter(|_| channels == 2);
            this.crossfeed = this.local_crossfeed.map(|(f, d)| Crossfeed::new(sr, f, d));
            if let Some(fir) = state.fir.as_ref() {
                this.local_fir = Arc::as_ptr(fir) as usize;
                this.conv = Some(super::fir::Convolver::new(fir, sr, channels.max(1) as usize));
            }
            if state.enabled && !state.bands.is_empty() {
                let active: Vec<&EqBand> = state.bands.iter().filter(|b| b.enabled).collect();
                this.filters_l = active.iter().map(|b| { let (bv, av) = eq_biquad_coeffs(b, sr); Biquad::new(bv, av) }).collect();
                this.filters_r = active.iter().map(|b| { let (bv, av) = eq_biquad_coeffs(b, sr); Biquad::new(bv, av) }).collect();
            }
        }
        this
    }

    /// Try to sync filter state from shared EQ. Uses try_lock so the audio
    /// thread NEVER blocks — if the UI holds the mutex, we skip this cycle
    /// and retry 512 samples later (~11 ms). Imperceptible for EQ changes.
    fn sync_filters(&mut self) {
        let Ok(eq) = self.eq.try_lock() else { return; };
        if eq.version == self.local_version { return; } // nothing changed
        self.local_version = eq.version;
        self.local_enabled = eq.enabled;
        self.gain = 10f64.powf(eq.preamp_db as f64 / 20.0);
        let want = eq.crossfeed.filter(|_| self.channels == 2);
        if want.is_none() {
            self.crossfeed = None;
        } else if self.crossfeed.is_none() || self.local_crossfeed != want {
            self.crossfeed = want.map(|(f, d)| Crossfeed::new(eq.sample_rate, f, d));
        }
        self.local_crossfeed = want;
        // A new room correction is built here, once, on the audio thread:
        // a user's choice, not a per-buffer cost.
        let fir = eq.fir.clone();
        let addr = fir.as_ref().map_or(0, |f| Arc::as_ptr(f) as usize);
        if addr != self.local_fir {
            self.local_fir = addr;
            self.conv = fir.map(|f| super::fir::Convolver::new(&f, eq.sample_rate, self.channels.max(1) as usize));
        }
        if !eq.enabled || eq.bands.is_empty() {
            self.filters_l.clear();
            self.filters_r.clear();
            return;
        }
        let sr = eq.sample_rate;
        let active: Vec<&EqBand> = eq.bands.iter().filter(|b| b.enabled).collect();
        self.filters_l = active.iter().map(|b| { let (bv, av) = eq_biquad_coeffs(b, sr); Biquad::new(bv, av) }).collect();
        self.filters_r = active.iter().map(|b| { let (bv, av) = eq_biquad_coeffs(b, sr); Biquad::new(bv, av) }).collect();
    }

    #[inline]
    fn apply(&mut self, sample: f32, ch: u16) -> f32 {
        if !self.local_enabled && self.conv.is_none() { return sample; }
        let mut s = sample as f64;
        if self.local_enabled {
            let filters = if ch == 0 { &mut self.filters_l } else { &mut self.filters_r };
            s *= self.gain;
            for f in filters.iter_mut() { s = f.process(s); }
        }
        if let Some(cv) = self.conv.as_mut() {
            s = cv.process(ch as usize, s as f32) as f64;
        }
        s.clamp(-1.0, 1.0) as f32
    }
}

impl<S> Iterator for EqSource<S>
where S: rodio::Source<Item = f32>
{
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        // Crossfeed needs both samples of a frame: the left comes out now,
        // the right is held for the next call.
        if let Some(r) = self.held.take() {
            self.ch_idx = 0;
            return Some(r);
        }
        if self.crossfeed.is_some() && self.ch_idx == 0 {
            let l = self.inner.next()?;
            let Some(r) = self.inner.next() else { return Some(l) };
            self.check_counter += 2;
            if self.check_counter >= 512 {
                self.check_counter = 0;
                self.sync_filters();
            }
            let (l, r) = (self.apply(l, 0) as f64, self.apply(r, 1) as f64);
            let (l, r) = match self.crossfeed.as_mut() {
                Some(cf) => cf.frame(l, r),
                None => (l, r),
            };
            self.held = Some(r.clamp(-1.0, 1.0) as f32);
            self.ch_idx = 1;
            return Some(l.clamp(-1.0, 1.0) as f32);
        }
        let sample = self.inner.next()?;
        // Poll for EQ changes every 512 samples (~11 ms at 44.1 kHz).
        // sync_filters uses try_lock — the audio thread NEVER blocks on the
        // UI mutex, so this cannot cause buffer underruns or noise.
        self.check_counter += 1;
        if self.check_counter >= 512 {
            self.check_counter = 0;
            self.sync_filters();
        }
        let ch = self.ch_idx;
        let out = self.apply(sample, ch);
        self.ch_idx = (self.ch_idx + 1) % self.channels.max(1);
        Some(out)
    }
}

impl<S> rodio::Source for EqSource<S>
where S: rodio::Source<Item = f32>
{
    fn current_frame_len(&self) -> Option<usize> { self.inner.current_frame_len() }
    fn channels(&self)         -> u16            { self.inner.channels() }
    fn sample_rate(&self)      -> u32            { self.inner.sample_rate() }
    fn total_duration(&self)   -> Option<Duration> { self.inner.total_duration() }
    fn try_seek(&mut self, pos: Duration) -> Result<(), rodio::source::SeekError> {
        self.inner.try_seek(pos)
    }
}

/// Direct-Form-II transposed biquad. Uses f64 internally for numerical stability.
pub struct Biquad { pub b: [f64; 3], pub a: [f64; 2], pub z1: f64, pub z2: f64 }
impl Biquad {
    pub fn new(b: [f64; 3], a: [f64; 2]) -> Self { Self { b, a, z1: 0.0, z2: 0.0 } }
    pub fn process(&mut self, x: f64) -> f64 {
        let y = self.b[0] * x + self.z1;
        self.z1 = self.b[1] * x - self.a[0] * y + self.z2;
        self.z2 = self.b[2] * x - self.a[1] * y;
        y
    }
}

/// ITU-R BS.1770-4 K-weighting filter (two biquad stages) for any sample rate.
pub struct KWeightFilter { pub stage1: Biquad, pub stage2: Biquad }
impl KWeightFilter {
    pub fn new(sr: u32) -> Self {
        use std::f64::consts::PI;
        let sr = sr as f64;
        // Stage 1: high-shelf pre-filter  (f0=1681.97 Hz, Q=0.7072, +4 dB shelf)
        let f0 = 1681.974450955533_f64;
        let q  = 0.7071752369554196_f64;
        let vh = 10_f64.powf(3.999843853973347 / 20.0); // linear shelf gain
        let vb = vh.sqrt();
        let k  = (PI * f0 / sr).tan();
        let d  = 1.0 + k / q + k * k;
        let b1 = [(vh + vb / q * k + k * k) / d,
                   2.0 * (k * k - vh) / d,
                   (vh - vb / q * k + k * k) / d];
        let a1 = [2.0 * (k * k - 1.0) / d, (1.0 - k / q + k * k) / d];
        // Stage 2: high-pass RLB weighting (f0=38.14 Hz, Q=0.5003)
        let f0 = 38.13547087602444_f64;
        let q  = 0.5003270373238773_f64;
        let k  = (PI * f0 / sr).tan();
        let d  = 1.0 + k / q + k * k;
        let b2 = [1.0 / d, -2.0 / d, 1.0 / d];
        let a2 = [2.0 * (k * k - 1.0) / d, (1.0 - k / q + k * k) / d];
        Self { stage1: Biquad::new(b1, a1), stage2: Biquad::new(b2, a2) }
    }
    pub fn process(&mut self, x: f32) -> f32 {
        self.stage2.process(self.stage1.process(x as f64)) as f32
    }
}

pub(crate) fn home_dir() -> PathBuf {
    #[cfg(test)]
    return super::test_profile::root();
    #[cfg(not(test))]
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
mod profile_tests {
    use super::*;

    /// The suite runs on the owner's machine. A library loaded in a test starts
    /// empty whatever the profile holds, and nothing a test does to one reaches
    /// the profile — the real file is compared before and after (read only).
    #[test]
    fn tests_never_read_or_write_the_owners_presets() {
        let real = EqPresetLibrary::preset_path();
        let before = std::fs::read(&real).ok();
        let lib = EqPresetLibrary::load();
        assert!(lib.default_id.is_none() && lib.global.is_empty() && lib.last_used.is_empty());
        let l = EqPresetLibrary {
            default_id: Some(7),
            last_used: [("C:/music/x.flac".to_string(), PresetRef { scope: PresetScope::Global, id: 7 })]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        l.save();
        assert_eq!(std::fs::read(&real).ok(), before, "a test wrote the owner's presets");
    }

    /// A shelf given by its Q (AutoEq's `autoeq/peq.py`, Equalizer APO's
    /// LSC/HSC: alpha = sin(w0) / 2Q) is, through `shelf_slope`, that very
    /// filter in Moosik, down to Q 0.01; taking the Q as the slope, as 1.6.2's
    /// first builds did, was off by up to 0.77 dB, and flooring the slope at
    /// 0.01 put a Q 0.05 shelf at +6 dB off by up to 0.47 dB.
    #[test]
    fn a_shelf_given_by_its_q_is_that_filter() {
        let by_q = |high: bool, fc: f64, gain: f64, q: f64, sr: u32| -> ([f64; 3], [f64; 2]) {
            let a = 10f64.powf(gain / 40.0);
            let w0 = std::f64::consts::TAU * fc / sr as f64;
            let (c, al, s) = (w0.cos(), w0.sin() / (2.0 * q), if high { -1.0 } else { 1.0 });
            let a0 = (a + 1.0) + s * (a - 1.0) * c + 2.0 * a.sqrt() * al;
            ([a * ((a + 1.0) - s * (a - 1.0) * c + 2.0 * a.sqrt() * al) / a0,
              s * 2.0 * a * ((a - 1.0) - s * (a + 1.0) * c) / a0,
              a * ((a + 1.0) - s * (a - 1.0) * c - 2.0 * a.sqrt() * al) / a0],
             [-s * 2.0 * ((a - 1.0) + s * (a + 1.0) * c) / a0,
              ((a + 1.0) + s * (a - 1.0) * c - 2.0 * a.sqrt() * al) / a0])
        };
        let worst = |b: &super::EqBand, (bq, aq): ([f64; 3], [f64; 2]), sr: u32| -> f32 {
            let (bm, am) = super::eq_biquad_coeffs(b, sr);
            (0..=400).map(|i| 20.0 * 1000f32.powf(i as f32 / 400.0))
                .map(|f| (super::biquad_response_db(bm, am, f, sr) - super::biquad_response_db(bq, aq, f, sr)).abs())
                .fold(0.0, f32::max)
        };
        let mut was = 0.0f32;
        for (high, fc, gain, q) in [(false, 105.0, 6.4, 0.7), (false, 105.0, 9.5, 0.7), (false, 105.0, -6.0, 0.7),
                                    (true, 10_000.0, -2.3, 0.7), (true, 8_000.0, 4.0, 0.5), (false, 200.0, 3.0, 0.4),
                                    (false, 105.0, 6.0, 0.05), (false, 60.0, 20.0, 0.02), (true, 3_000.0, -12.0, 0.01)] {
            for sr in [44_100, 48_000, 96_000] {
                let kind = if high { super::BandKind::HighShelf } else { super::BandKind::LowShelf };
                let band = |q| super::EqBand { freq: fc as f32, gain_db: gain as f32, q, kind: kind.clone(), enabled: true };
                let want = by_q(high, fc as f32 as f64, gain as f32 as f64, q, sr);
                let now = worst(&band(super::shelf_slope(q, gain as f32)), want, sr);
                assert!(now < 1e-4, "{fc} Hz {gain:+} dB Q {q} at {sr}: off by {now} dB");
                was = was.max(worst(&band(q as f32), want, sr));
            }
        }
        assert!(was > 0.7, "the Q taken as the slope was off by {was} dB");
    }

    /// Above a slope of 0.01 (the band table offers 0.1 to 10), a shelf's
    /// coefficients are what they were before the slope stopped being
    /// floored, to the bit: presets keep their sound.
    #[test]
    fn a_shelf_above_slope_0_01_is_as_it_was() {
        let old = |b: &super::EqBand, sr: u32| -> [f64; 5] {
            let sr = sr as f64;
            let f0 = (b.freq as f64).clamp(1.0, sr * 0.499);
            let q = (b.q as f64).max(0.01);
            let w0 = 2.0 * std::f64::consts::PI * f0 / sr;
            let (cw, sw, a) = (w0.cos(), w0.sin(), 10f64.powf(b.gain_db as f64 / 40.0));
            let as_ = (sw / 2.0) * ((a + 1.0/a) * (1.0/q - 1.0) + 2.0).max(0.0).sqrt();
            let (b0, b1, b2, a0, a1, a2) = if b.kind == super::BandKind::LowShelf {
                (a*((a+1.0)-(a-1.0)*cw+2.0*a.sqrt()*as_), 2.0*a*((a-1.0)-(a+1.0)*cw), a*((a+1.0)-(a-1.0)*cw-2.0*a.sqrt()*as_),
                 (a+1.0)+(a-1.0)*cw+2.0*a.sqrt()*as_, -2.0*((a-1.0)+(a+1.0)*cw), (a+1.0)+(a-1.0)*cw-2.0*a.sqrt()*as_)
            } else {
                (a*((a+1.0)+(a-1.0)*cw+2.0*a.sqrt()*as_), -2.0*a*((a-1.0)+(a+1.0)*cw), a*((a+1.0)+(a-1.0)*cw-2.0*a.sqrt()*as_),
                 (a+1.0)-(a-1.0)*cw+2.0*a.sqrt()*as_, 2.0*((a-1.0)-(a+1.0)*cw), (a+1.0)-(a-1.0)*cw-2.0*a.sqrt()*as_)
            };
            [b0/a0, b1/a0, b2/a0, a1/a0, a2/a0]
        };
        let mut n = 0;
        for kind in [super::BandKind::LowShelf, super::BandKind::HighShelf] {
            for q in [0.0101f32, 0.02, 0.1, 0.3, std::f32::consts::FRAC_1_SQRT_2, 1.0, 1.5, 3.0, 10.0] {
                for gain_db in [-40.0f32, -24.0, -6.0, -0.5, 0.0, 0.5, 3.0, 12.0, 24.0, 40.0] {
                    for freq in [20.0f32, 105.0, 1000.0, 10_000.0, 21_000.0] {
                        for sr in [44_100, 48_000, 96_000, 192_000] {
                            let b = super::EqBand { freq, gain_db, q, kind: kind.clone(), enabled: true };
                            let (bn, an) = super::eq_biquad_coeffs(&b, sr);
                            let now = [bn[0], bn[1], bn[2], an[0], an[1]];
                            assert!(now.iter().zip(old(&b, sr)).all(|(x, y)| x.to_bits() == y.to_bits()), "slope {q}, {gain_db} dB, {freq} Hz at {sr}");
                            n += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(n, 3600);
    }

    #[test]
    fn an_autoeq_profile_becomes_a_preamp_and_bands() {
        let text = concat!("Preamp: -6.4 dB", "\r\n", "Filter 1: ON LSC Fc 105 Hz Gain 6.4 dB Q 0.70", "\r\n", "Filter 2: ON PK Fc 165 Hz Gain -3.1 dB Q 0.42", "\r\n", "Filter 3: OFF PK Fc 3000 Hz Gain 2.0 dB Q 1.2", "\r\n", "Filter 10: ON HSC Fc 10000 Hz Gain -2.3 dB Q 0.70", "\r\n");
        let (pre, bands) = super::parse_autoeq(text).unwrap();
        assert_eq!(pre, -6.4);
        assert_eq!(bands.len(), 4);
        assert!(bands[0].kind == super::BandKind::LowShelf && bands[0].freq == 105.0 && bands[0].gain_db == 6.4);
        assert_eq!(bands[0].q, super::shelf_slope(0.70f32 as f64, 6.4), "a shelf's Q, as Moosik's slope");
        assert!(bands[1].kind == super::BandKind::Peaking && bands[1].gain_db == -3.1 && bands[1].q == 0.42);
        assert!(!bands[2].enabled, "OFF is kept, switched off");
        assert!(bands[3].kind == super::BandKind::HighShelf);
        // Equalizer APO's unnumbered form, and a filter type Moosik lacks.
        assert_eq!(super::parse_autoeq("Filter: ON PK Fc 1000 Hz Gain 3 dB Q 2").unwrap().1.len(), 1);
        assert!(super::parse_autoeq("Filter 1: ON BP Fc 1000 Hz").is_err());
        // A gentle shelf is kept as given; a filter below Q 0.01 is refused, not changed.
        assert_eq!(super::parse_autoeq("Filter: ON LSC Fc 105 Hz Gain 6 dB Q 0.05").unwrap().1[0].q, super::shelf_slope(0.05f32 as f64, 6.0));
        assert!(super::parse_autoeq("Filter: ON PK Fc 1000 Hz Gain 3 dB Q 0.005").is_err_and(|e| e.contains("Q 0.01")));
        assert!(super::parse_autoeq("hello").is_err());
        let many: String = (1..=MAX_BANDS + 1).map(|i| format!("Filter {i}: ON PK Fc {} Hz Gain 1 dB Q 1\n", 100 * i)).collect();
        assert!(super::parse_autoeq(&many).is_err());
    }

    #[test]
    fn the_preamp_scales_what_is_heard() {
        let mut st = super::EqState::new();
        st.preamp_db = -6.0206;
        assert!(st.is_active(), "a preamp alone puts the EQ in the path");
        let eq = std::sync::Arc::new(std::sync::Mutex::new(st));
        let src = rodio::buffer::SamplesBuffer::new(2, 48_000, vec![0.5f32; 2000]);
        let out: Vec<f32> = super::EqSource::new(src, eq).collect();
        assert_eq!(out.len(), 2000);
        assert!(out.iter().all(|v| (v - 0.25).abs() < 1e-4), "{}", out[0]);
    }

    #[test]
    fn crossfeed_leaks_a_quieter_low_passed_copy_to_the_other_ear() {
        let mut cf = super::Crossfeed::new(48_000, 700.0, 4.5);
        // Steady left-only signal: the right ear hears it 4.5 dB below the left.
        let (mut l, mut r) = (0.0, 0.0);
        for _ in 0..48_000 {
            (l, r) = cf.frame(0.5, 0.0);
        }
        let feed = 20.0 * (l / r).log10();
        assert!((feed - 4.5).abs() < 0.1, "{feed}");
        // A 10 kHz tone in the left: almost nothing reaches the right.
        let mut cf = super::Crossfeed::new(48_000, 700.0, 4.5);
        let (mut pl, mut pr) = (0f64, 0f64);
        for i in 0..48_000 {
            let x = (2.0 * std::f64::consts::PI * 10_000.0 * i as f64 / 48_000.0).sin() * 0.5;
            let (a, b) = cf.frame(x, 0.0);
            if i > 4800 {
                pl = pl.max(a.abs());
                pr = pr.max(b.abs());
            }
        }
        assert!(20.0 * (pl / pr).log10() > 20.0, "{pl} {pr}");
    }

    #[test]
    fn crossfeed_runs_frame_by_frame_in_the_eq_path() {
        let mut st = super::EqState::new();
        st.crossfeed = Some((700.0, 4.5));
        let eq = std::sync::Arc::new(std::sync::Mutex::new(st));
        // Left only, steady: the right channel fills in.
        let src = rodio::buffer::SamplesBuffer::new(2, 48_000, (0..96_000).map(|i| if i % 2 == 0 { 0.5f32 } else { 0.0 }).collect::<Vec<_>>());
        let out: Vec<f32> = super::EqSource::new(src, eq).collect();
        assert_eq!(out.len(), 96_000);
        let (l, r) = (out[95_998], out[95_999]);
        assert!(r > 0.1 && l > r, "{l} {r}");
    }

    #[test]
    fn crossfeed_stays_on_when_the_eq_is_switched_off() {
        let mut st = super::EqState::new();
        st.enabled = false;
        st.preamp_db = -20.0;
        st.crossfeed = Some((700.0, 4.5));
        assert!(st.is_active());
        let eq = std::sync::Arc::new(std::sync::Mutex::new(st));
        let src = rodio::buffer::SamplesBuffer::new(2, 48_000, (0..96_000).map(|i| if i % 2 == 0 { 0.5f32 } else { 0.0 }).collect::<Vec<_>>());
        let out: Vec<f32> = super::EqSource::new(src, eq).collect();
        let (l, r) = (out[95_998], out[95_999]);
        assert!(r > 0.1, "crossfeed still on: {r}");
        assert!(l > 0.3, "the EQ's preamp is off with the EQ: {l}");
    }
}

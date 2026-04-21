use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Parametric EQ — types, biquad math, shared state
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Serialize, Deserialize)]
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
    pub enabled:     bool,
    pub sample_rate: u32,
    /// Incremented on every change so EqSource can detect updates cheaply.
    pub version:     u64,
}

impl EqState {
    pub fn new() -> Self {
        Self { bands: Vec::new(), enabled: true, sample_rate: 44_100, version: 0 }
    }
    /// Call after mutating bands or sample_rate.
    pub fn bump(&mut self) { self.version = self.version.wrapping_add(1); }
    /// True when EQ is enabled and at least one band is active.
    pub fn is_active(&self) -> bool {
        self.enabled && self.bands.iter().any(|b| b.enabled)
    }
    /// Approximate fingerprint used for the bake-to-cache key.
    pub fn fingerprint(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        for b in &self.bands {
            if !b.enabled { continue; }
            ((b.freq   * 10.0) as i32).hash(&mut h);
            ((b.gain_db * 10.0) as i32).hash(&mut h);
            ((b.q      * 100.0) as i32).hash(&mut h);
            (b.kind.label()).hash(&mut h);
        }
        h.finish()
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
        std::fs::read_to_string(Self::preset_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
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
            let as_ = (sw / 2.0) * ((a + 1.0/a) * (1.0/q - 1.0) + 2.0).max(0.0).sqrt();
            ( a*((a+1.0)-(a-1.0)*cw+2.0*a.sqrt()*as_),
              2.0*a*((a-1.0)-(a+1.0)*cw),
              a*((a+1.0)-(a-1.0)*cw-2.0*a.sqrt()*as_),
              (a+1.0)+(a-1.0)*cw+2.0*a.sqrt()*as_,
              -2.0*((a-1.0)+(a+1.0)*cw),
              (a+1.0)+(a-1.0)*cw-2.0*a.sqrt()*as_ )
        }
        BandKind::HighShelf => {
            let a  = 10f64.powf(band.gain_db as f64 / 40.0);
            let as_ = (sw / 2.0) * ((a + 1.0/a) * (1.0/q - 1.0) + 2.0).max(0.0).sqrt();
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

/// Total EQ response in dB at `freq`, summing all enabled bands.
pub fn total_eq_response_db(bands: &[EqBand], sr: u32, freq: f32) -> f32 {
    bands.iter()
        .filter(|b| b.enabled)
        .map(|b| { let (b_a, a_a) = eq_biquad_coeffs(b, sr); biquad_response_db(b_a, a_a, freq, sr) })
        .sum()
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
        if !self.local_enabled { return sample; }
        let filters = if ch == 0 { &mut self.filters_l } else { &mut self.filters_r };
        let mut s = sample as f64;
        for f in filters.iter_mut() { s = f.process(s); }
        s.clamp(-1.0, 1.0) as f32
    }
}

impl<S> Iterator for EqSource<S>
where S: rodio::Source<Item = f32>
{
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
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
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

//! Adaptive Superlet Transform (ASLT) — Moca et al., *Nature Communications* 2021.
//!
//! A superlet is a set of Morlet wavelets that all sit at the same centre
//! frequency but differ in how many cycles they span. Short wavelets pin down
//! *when* something happened; long ones pin down *what pitch* it was. Taking the
//! geometric mean of their responses is a soft logical AND: a bar only lights up
//! where every member agrees there is energy, which localises far better than any
//! single window can.
//!
//! # What it buys over the FFT path
//!
//! The FFT path analyses every frequency through one window, ~171 ms wide, so
//! its resolution is a flat σ ≈ 3.6 Hz. A log bar grid does not want flat: it
//! wants σ = f/Q. Those two cross around 500 Hz, and that crossing decides
//! everything:
//!
//! * **Below it** the FFT runs out of resolution before the display does. At
//!   60 Hz the grid asks for 0.42 Hz and the FFT can only give 3.6 Hz, so eight
//!   bars out of every nine are drawing the same information. A superlet reaches
//!   the grid here — but only by using windows several seconds long, which is
//!   why `max_window_s` exists and why the bass visibly slows down.
//! * **Above it** the FFT is already finer than 1024 bars can render, so extra
//!   frequency resolution is invisible. The win there is time: at 10 kHz a
//!   grid-matched superlet needs 24 ms against the FFT's 171 ms.
//!
//! So the transform is not uniformly "better" — it trades time for frequency at
//! the bottom and frequency for time at the top, and the FFT is genuinely hard
//! to beat in the octave either side of the crossover.
//!
//! This module is deliberately standalone — it takes a signal and returns raw
//! magnitudes. dB conversion, ISO 226 weighting and caching all stay in
//! `spectrum.rs`, which keeps the transform testable in isolation.

// Phase 0 ships the transform standalone so it can be measured before it is
// wired in; nothing outside the tests calls it yet. The integration phase
// removes this.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Cycle count of the shortest wavelet in every superlet. 3 is the practical
/// floor for a Morlet — below it the wavelet stops resembling a wave packet and
/// its Gaussian envelope no longer suppresses the negative-frequency lobe.
pub const DEFAULT_C_MIN: f32 = 3.0;

/// Gaussian envelopes are truncated at this many standard deviations.
///
/// The cut leaves a step of `exp(-s²/2)` at the edge, which leaks as sidelobes:
/// 3σ → −39 dB, 4σ → −69 dB, 4.5σ → −88 dB. Cost is linear in this value.
///
/// 4.0 is the compromise: −69 dB sits under anything real music puts on screen,
/// while 4.5 cost 12 % more for headroom only a synthetic test tone would ever
/// use. It matters more than it looks — a Gaussian truncated at ±4σ needs about
/// twice the total length of a Hann window of equal main-lobe width, and that
/// factor is most of why a superlet has to run longer than the FFT it replaces.
const SUPPORT_SIGMAS: f32 = 4.0;

/// Magnitudes below this are treated as this before taking a logarithm.
///
/// The geometric mean is evaluated as `exp(mean(ln|R_i|))` rather than as a
/// literal product. A product of 48 magnitudes around 1e-10 underflows f32 long
/// before the root is taken; in log space there is nothing to underflow.
const MAG_FLOOR: f32 = 1e-30;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Adaptive superlet parameters.
///
/// # Why this is not the paper's `o_min`/`o_max`
///
/// The first cut followed Moca et al. directly: order `N` meant `N` wavelets
/// with cycles `3, 6, 9 … 3N`, ramped across the spectrum. Measured against the
/// FFT path it replaces, that was worse on both axes between 150 Hz and 2 kHz —
/// coarser in frequency *and* three times slower in time. Two reasons:
///
/// 1. Effective Q is `√(mean(cᵢ²))`, so a set running from 3 up to `c_max`
///    lands at `c_max/√3`. Time cost is set by the longest member alone. The
///    short members therefore cost nothing but drag resolution down 1.73×.
/// 2. Tying the *count* of wavelets to `c_max` meant 83 wavelets to reach the
///    display's Q. The disagreement-suppression that justifies a superlet
///    saturates after a handful; the other ~78 were pure overhead.
///
/// So the parameters here are the three things that actually matter, decoupled:
/// how sharp to be (`q_ratio`), how much time that is allowed to cost
/// (`max_window_s`), and how much of a spread to average over (`n_wavelets`,
/// `spread`). Cycle counts are then *solved for* rather than guessed.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct AsltConfig {
    /// Target effective Q as a multiple of the display grid's own Q.
    ///
    /// 1.0 resolves exactly what the bars can show; higher is invisible detail,
    /// lower means neighbouring bars read overlapping information.
    pub q_ratio: f32,
    /// Longest analysis window permitted, in seconds (full support, not σ).
    ///
    /// This is the honest bass knob. Constant-Q at 20 Hz mathematically requires
    /// a ~10 s window, so something has to give, and it should be a number the
    /// user sets rather than a curve buried in the code. Above roughly 1 kHz it
    /// never binds.
    pub max_window_s: f32,
    /// Wavelets per superlet. 1 is a plain Morlet CWT — sharpest possible peak,
    /// no cross-checking. More members suppress energy the set disagrees about,
    /// with diminishing returns past about 5.
    pub n_wavelets: usize,
    /// Shortest member's cycle count as a fraction of the longest.
    ///
    /// This is the dial the paper's formulation lacks. 0 reproduces it (and its
    /// 1.73× resolution penalty); 1 collapses to a single wavelet. Around 0.5
    /// keeps most of the cross-checking for a 1.31× penalty instead.
    pub spread: f32,
    /// How bar centre frequencies are distributed across the display.
    ///
    /// `Log` is what this has always done and stays the default; it takes the
    /// closed-form `grid_q` untouched, so nothing about an existing analysis
    /// changes. `Erb` spaces bars per auditory filter instead of per octave,
    /// which makes the Q target vary with frequency — see
    /// [`super::freq_scale`].
    #[serde(default)]
    pub scale: super::freq_scale::FreqScale,
}

impl Default for AsltConfig {
    fn default() -> Self { AsltPreset::Standard.config() }
}

/// Quality ladder. Each rung mostly buys *bass*: the window cap is what decides
/// how far below the FFT's fixed 171 ms the transform is allowed to reach.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum AsltPreset { Fast, Standard, High, Ultra, Extreme }

impl AsltPreset {
    pub fn config(self) -> AsltConfig {
        let (q_ratio, max_window_s, n_wavelets, spread) = match self {
            // Below FFT-equivalent everywhere, but very fast to compute and
            // still far better than the FFT in the treble's time response.
            AsltPreset::Fast     => (0.7, 0.25, 3, 0.5),
            // Roughly FFT parity in the bass, grid-exact from ~500 Hz up.
            AsltPreset::Standard => (1.0, 0.60, 4, 0.5),
            // First rung that clearly beats the FFT below 200 Hz.
            AsltPreset::High     => (1.0, 1.20, 5, 0.5),
            AsltPreset::Ultra    => (1.0, 2.50, 6, 0.55),
            // Grid-exact to ~35 Hz. Bass moves like treacle; that is what
            // constant-Q at the bottom of hearing actually costs.
            AsltPreset::Extreme  => (1.0, 5.00, 7, 0.6),
        };
        AsltConfig { q_ratio, max_window_s, n_wavelets, spread, scale: Default::default() }
    }

    pub fn label(self) -> &'static str {
        match self {
            AsltPreset::Fast     => "Fast",
            AsltPreset::Standard => "Standard",
            AsltPreset::High     => "High",
            AsltPreset::Ultra    => "Ultra",
            AsltPreset::Extreme  => "Extreme",
        }
    }
}

// ---------------------------------------------------------------------------
// Order / frequency geometry
// ---------------------------------------------------------------------------

/// Ratio of a superlet's effective Q to its longest member's cycle count.
///
/// Members are spread uniformly over `[spread·c_max, c_max]`, and the geometric
/// mean averages inverse variances, so `Q_eff = √(mean(cᵢ²))`. For a continuous
/// spread that is `c_max·√((1+α+α²)/3)`: 0.577 at `α=0` (the paper's layout),
/// 0.764 at `α=0.5`, 1.0 at `α=1`. Time cost is set by `c_max` alone, so this
/// factor is pure efficiency — how much resolution the set keeps for what it
/// spends.
fn q_per_cycle(cfg: &AsltConfig) -> f32 {
    let a = cfg.spread.clamp(0.0, 1.0);
    let n = cfg.n_wavelets.max(1);
    if n == 1 { return 1.0; }
    // Discrete mean, so the reported figure matches the wavelets actually built.
    let mean_sq: f32 = (0..n)
        .map(|i| {
            let frac = a + (1.0 - a) * i as f32 / (n - 1) as f32;
            frac * frac
        })
        .sum::<f32>() / n as f32;
    mean_sq.sqrt()
}

/// Effective Q the transform will actually deliver at `freq`.
///
/// Two ceilings, whichever bites first:
/// * the display grid (`grid_q · q_ratio`) — beyond it the extra sharpness has
///   nowhere to appear, since neighbouring bars are already distinct;
/// * the window budget — constant-Q at 20 Hz wants a ~10 s window, so below
///   some frequency the clock wins and resolution has to give.
///
/// The crossover is the whole story of this transform. Above it the superlet
/// runs *shorter* windows than the FFT for the same visible detail; below it,
/// longer ones for detail the FFT cannot reach at any setting.
pub fn effective_q_at(freq: f32, grid_q: f32, cfg: &AsltConfig) -> f32 {
    let target = grid_q * cfg.q_ratio.max(0.01);
    // support = 2·SUPPORT_SIGMAS·σ_t and σ_t = c_max/(2πf), so
    // c_max ≤ π·f·window / SUPPORT_SIGMAS.
    let c_cap = std::f32::consts::PI * freq * cfg.max_window_s.max(0.001)
        / SUPPORT_SIGMAS;
    let q_cap = c_cap * q_per_cycle(cfg);
    target.min(q_cap).max(1.0)
}

/// Longest member's cycle count needed to reach `q` — the inverse of
/// [`q_per_cycle`]. This is what sets the window length and therefore the cost.
fn c_max_for_q(q: f32, cfg: &AsltConfig) -> f32 {
    (q / q_per_cycle(cfg).max(1e-6)).max(DEFAULT_C_MIN)
}

/// Lowest frequency that still reaches the full bar-grid resolution.
///
/// This is what `max_window_s` is really choosing, stated in the units the
/// choice is actually about. Above this frequency a tone lands in one bar;
/// below it, the window cap binds and the peak spreads over roughly
/// `grid_q / effective_q_at(f)` bars.
///
/// Inverting `effective_q_at`'s cap: `q_cap = π·f·T·q_per_cycle / SUPPORT_SIGMAS`,
/// so setting `q_cap` equal to the grid target gives
/// `f = SUPPORT_SIGMAS·grid_q·q_ratio / (π·T·q_per_cycle)`.
pub fn full_detail_above(grid_q: f32, cfg: &AsltConfig) -> f32 {
    let qpc = q_per_cycle(cfg).max(1e-6);
    SUPPORT_SIGMAS * grid_q * cfg.q_ratio.max(0.01)
        / (std::f32::consts::PI * cfg.max_window_s.max(0.001) * qpc)
}

/// The window length that puts full grid resolution down at `freq` — the
/// inverse of [`full_detail_above`], so the UI can offer the choice either way
/// round.
pub fn window_for_full_detail_above(freq: f32, grid_q: f32, cfg: &AsltConfig) -> f32 {
    let qpc = q_per_cycle(cfg).max(1e-6);
    SUPPORT_SIGMAS * grid_q * cfg.q_ratio.max(0.01)
        / (std::f32::consts::PI * freq.max(1.0) * qpc)
}

/// Full analysis window at `freq`, in seconds — what the bass actually smears
/// over. Quoted in the UI because it is the cost the user feels.
pub fn window_seconds_at(freq: f32, grid_q: f32, cfg: &AsltConfig) -> f32 {
    let c_max = c_max_for_q(effective_q_at(freq, grid_q, cfg), cfg);
    2.0 * SUPPORT_SIGMAS * c_max / (std::f32::consts::TAU * freq)
}

/// Log-spaced centre frequency of bar `i` of `n`. Mirrors `bar_center_freq` in
/// `spectrum.rs` so both paths land on identical bars.
pub fn bar_center_freq(i: usize, n: usize, min_freq: f32, max_freq: f32) -> f32 {
    let t = (i as f32 + 0.5) / n as f32;
    10f32.powf(min_freq.log10() + t * (max_freq.log10() - min_freq.log10()))
}

/// Temporal standard deviation, in samples, of a `cycles`-cycle Morlet at `freq`.
fn sigma_samples(freq: f32, cycles: f32, sample_rate: f32) -> f32 {
    cycles * sample_rate / (std::f32::consts::TAU * freq)
}

/// Effective Q of a whole superlet: the response to a tone `Δ` Hz off centre
/// falls as `exp(-Δ²/2σ²)` with `σ = freq / effective_q(order, c_min)`.
///
/// Worth stating explicitly because the intuitive answer is wrong. A superlet is
/// *not* as sharp as its narrowest member. Member `i` has `σ_i = freq/(i·c_min)`,
/// and the geometric mean multiplies Gaussians, which averages their inverse
/// variances rather than picking the smallest:
///
/// ```text
/// 1/σ_eff² = (1/N)·Σ (i·c_min/freq)²   ⇒   Q_eff = c_min·√((N+1)(2N+1)/6)
/// ```
///
/// So order 20 at `c_min = 3` gives `Q_eff ≈ 36`, not the 60 its longest wavelet
/// alone would manage. The wide members are what suppress spurious energy — that
/// is the point of the transform — but they cost main-lobe width to do it.
///
/// Practical consequence: matching a 1024-bar log grid (`Q ≈ 144`) needs order
/// ≈ 82, not 48. That is what this function is for — quoting the honest
/// resolution of a preset instead of its headline cycle count.
/// The Q a log-spaced bar grid implies: adjacent bars are `f/q` apart, so a
/// transform with a lower effective Q has neighbouring bars reading overlapping
/// information and the extra bars show nothing new.
pub fn grid_q(n_bars: usize, min_freq: f32, max_freq: f32) -> f32 {
    let octaves = (max_freq / min_freq).log2().max(0.1);
    let per_octave = n_bars as f32 / octaves;
    1.0 / (2f32.powf(1.0 / per_octave) - 1.0)
}

/// Frequency resolution of the FFT path this transform competes with, as a
/// Gaussian-equivalent σ in Hz.
///
/// `auto_fft_size_for` keeps the analysis window at 100–200 ms, and a Hann
/// window's −3 dB main lobe is 1.44 bins wide, so σ ≈ 1.44·(SR/N)/2.355.
/// Constant across the spectrum, which is exactly why a constant-Q display can
/// beat it at the bottom and never needs to at the top.
pub fn fft_sigma_hz(sample_rate: u32, fft_size: usize) -> f32 {
    1.44 * (sample_rate as f32 / fft_size.max(1) as f32) / 2.355
}

/// Frequency below which the FFT runs out of resolution before the bar grid
/// does — the region where a superlet can show detail no FFT setting reaches.
/// Above it the FFT is already finer than the display can render, so the only
/// gain available is a shorter window.
pub fn fft_crossover_hz(sample_rate: u32, fft_size: usize, grid_q: f32) -> f32 {
    grid_q * fft_sigma_hz(sample_rate, fft_size)
}

// ---------------------------------------------------------------------------
// Morlet kernel
// ---------------------------------------------------------------------------

/// A complex Morlet, pre-multiplied into real and imaginary tables so the inner
/// loop is two multiply-accumulates and no trigonometry.
pub struct Morlet {
    /// `g(k)·cos(ωk)`, indexed from `-half` to `+half`.
    re: Vec<f32>,
    /// `-g(k)·sin(ωk)`.
    im: Vec<f32>,
    /// The bare normalised envelope `g(k)`, summing to 1 across the full table.
    /// Only needed at the signal edges, where the sum over the surviving taps
    /// becomes the divisor (see [`Morlet::response`]).
    env: Vec<f32>,
    half: usize,
}

impl Morlet {
    /// Build the wavelet for `freq` spanning `cycles` cycles.
    ///
    /// The Gaussian envelope is L1-normalised, which makes the response to a
    /// unit-amplitude sine equal 1.0 regardless of frequency or cycle count
    /// (see `response`). Without that, changing preset would rescale the whole
    /// display and shift every bar against the ISO 226 weights.
    pub fn new(freq: f32, cycles: f32, sample_rate: f32) -> Self {
        let sigma = sigma_samples(freq, cycles, sample_rate).max(0.5);
        let half = ((SUPPORT_SIGMAS * sigma).ceil() as usize).max(1);
        let n = 2 * half + 1;

        let mut env = Vec::with_capacity(n);
        let mut sum = 0.0f64;
        for k in 0..n {
            let t = k as f32 - half as f32;
            let g = (-(t * t) / (2.0 * sigma * sigma)).exp();
            sum += g as f64;
            env.push(g);
        }

        let norm = if sum > 0.0 { (1.0 / sum) as f32 } else { 0.0 };
        let omega = std::f32::consts::TAU * freq / sample_rate;
        let mut re = Vec::with_capacity(n);
        let mut im = Vec::with_capacity(n);
        let mut env_n = Vec::with_capacity(n);
        for (k, g) in env.into_iter().enumerate() {
            let t = k as f32 - half as f32;
            let phase = omega * t;
            let g = g * norm;
            re.push(g * phase.cos());
            im.push(-g * phase.sin());
            env_n.push(g);
        }

        Self { re, im, env: env_n, half }
    }

    /// Magnitude of the wavelet response centred on sample `centre`.
    ///
    /// Returns 1.0 for a unit-amplitude sine at the wavelet's own frequency: the
    /// analytic response to `cos` is half the L1 envelope sum, hence the ×2.
    ///
    /// **Edges are truncated, not padded.** Taps falling outside the signal are
    /// dropped and the envelope is renormalised over the survivors, which is
    /// equivalent to analysing with a correctly-scaled half-Gaussian. Padding was
    /// tried and is worse either way: zeros cost a 6 dB dip that takes a full
    /// support-width to recover from, and mirroring is actively wrong for
    /// oscillatory content — an even reflection of a sine cancels against the
    /// wavelet's odd component and collapses the reading to near zero. Truncation
    /// invents no samples; it just loses frequency resolution near the boundary,
    /// which is the honest trade.
    ///
    /// Accumulates in f64. A 4.5σ wavelet at 20 Hz spans ~600 k taps, and an f32
    /// accumulator drifts measurably over that many adds.
    fn response(&self, signal: &[f32], centre: isize) -> f32 {
        let n = signal.len() as isize;
        let base = centre - self.half as isize;
        let end = base + self.re.len() as isize;
        let lo = base.max(0);
        let hi = end.min(n);
        if hi <= lo { return 0.0; }

        let mut acc_re = 0.0f64;
        let mut acc_im = 0.0f64;

        if base >= 0 && end <= n {
            // Interior: the envelope already sums to 1, so skip that accumulator.
            for (k, (&wr, &wi)) in self.re.iter().zip(self.im.iter()).enumerate() {
                let s = signal[(base + k as isize) as usize];
                acc_re += (s * wr) as f64;
                acc_im += (s * wi) as f64;
            }
            return (2.0 * (acc_re * acc_re + acc_im * acc_im).sqrt()) as f32;
        }

        let mut env_sum = 0.0f64;
        for i in lo..hi {
            let k = (i - base) as usize;
            let s = signal[i as usize];
            acc_re += (s * self.re[k]) as f64;
            acc_im += (s * self.im[k]) as f64;
            env_sum += self.env[k] as f64;
        }
        if env_sum <= 0.0 { return 0.0; }
        (2.0 * (acc_re * acc_re + acc_im * acc_im).sqrt() / env_sum) as f32
    }

    /// Magnitude at every hop-spaced frame, by whichever route is cheaper.
    ///
    /// Both produce the same numbers; see [`Morlet::magnitudes_via_fft`] for why
    /// there are two.
    fn magnitudes(&self, signal: &[f32], hop: usize, frames: usize) -> Vec<f32> {
        if fft_is_cheaper(self.re.len(), hop)
            && let Some(v) = self.magnitudes_via_fft(signal, hop, frames)
        {
            return v;
        }
        (0..frames).map(|fi| self.response(signal, (fi * hop) as isize)).collect()
    }

    /// Interior frames by overlap-save FFT convolution; edge frames by the
    /// direct loop.
    ///
    /// Splitting them is not an optimisation, it is what keeps the two routes
    /// identical. A frame whose window hangs off the end of the signal has its
    /// envelope renormalised over the taps that survive (see [`Morlet::response`]),
    /// and a convolution cannot express that — it would silently reintroduce the
    /// zero-padding dip this transform was written to avoid. Edge frames are a
    /// few percent of a track, so keeping them on the direct path costs almost
    /// nothing.
    ///
    /// Returns `None` when the signal is shorter than the kernel, in which case
    /// every frame is an edge frame anyway.
    /// Tap count — what the GPU pipeline needs to size its buffers.
    pub fn taps(&self) -> usize { self.re.len() }
    pub fn re_taps(&self) -> &[f32] { &self.re }
    pub fn im_taps(&self) -> &[f32] { &self.im }
    pub fn half_width(&self) -> usize { self.half }

    /// The shared-signal route, exposed so the GPU pipeline can be held to it
    /// and so the calibration probe can time the two against each other.
    pub fn magnitudes_via_shared_for_test(
        &self, signal: &[f32], blocks: &SignalBlocks, hop: usize, frames: usize,
    ) -> Option<Vec<f32>> {
        self.magnitudes_via_shared(signal, blocks, hop, frames)
    }

    /// As [`Morlet::magnitudes_via_fft`], but reusing a signal transform that
    /// was computed once for every kernel of this block size.
    ///
    /// The only structural difference is where the blocks come from and that
    /// the stride is the shared `n/2` rather than this kernel's own
    /// `n - k + 1`. Everything after the pointwise multiply — which outputs are
    /// valid, how they map to frames, how edge frames are handled — is
    /// identical, which is what lets the route-equivalence tests cover it.
    fn magnitudes_via_shared(
        &self, signal: &[f32], blocks: &SignalBlocks, hop: usize, frames: usize,
    ) -> Option<Vec<f32>> {
        use rustfft::num_complex::Complex;

        let k = self.re.len();
        let n_sig = signal.len();
        let half = self.half as isize;
        let n = blocks.n;
        if n_sig <= k || hop == 0 || frames == 0 || n <= k { return None; }
        // A shorter stride than this would leave gaps between valid regions.
        if blocks.stride > n - k + 1 { return None; }

        let first = (half as usize).div_ceil(hop);
        let last_pos = (n_sig as isize) - 1 - half;
        if last_pos < 0 { return None; }
        let last = ((last_pos as usize) / hop).min(frames.saturating_sub(1));
        if first > last { return None; }

        let inv = ASLT_PLANNER.with(|p| p.borrow_mut().plan_fft_inverse(n));

        let mut kernel = vec![Complex { re: 0.0f32, im: 0.0f32 }; n];
        for (slot, i) in kernel.iter_mut().zip((0..k).rev()) {
            *slot = Complex { re: self.re[i], im: self.im[i] };
        }
        ASLT_PLANNER.with(|p| p.borrow_mut().plan_fft_forward(n)).process(&mut kernel);

        let mut out = vec![0.0f32; frames];
        for (fi, slot) in out.iter_mut().enumerate() {
            if fi < first || fi > last {
                *slot = self.response(signal, (fi * hop) as isize);
            }
        }

        let scale = 1.0 / n as f32;
        let mut buf = vec![Complex { re: 0.0f32, im: 0.0f32 }; n];
        for b in 0..blocks.blocks() {
            let base = b * blocks.stride;
            if base >= n_sig { break; }
            buf.copy_from_slice(blocks.block(b));
            for (x, h) in buf.iter_mut().zip(kernel.iter()) { *x *= *h; }
            inv.process(&mut buf);

            let m_lo = base + k - 1;
            let m_hi = (base + n).min(n_sig);
            let fi_lo = ((m_lo as isize - half).max(0) as usize).div_ceil(hop).max(first);
            let fi_hi = if (m_hi as isize) - 1 - half < 0 {
                0
            } else {
                ((((m_hi as isize) - 1 - half) as usize) / hop).min(last)
            };
            if fi_lo <= fi_hi {
                for (fi, slot) in (fi_lo..=fi_hi).zip(out[fi_lo..=fi_hi].iter_mut()) {
                    let c = buf[fi * hop + half as usize - base];
                    *slot = 2.0 * scale * (c.re * c.re + c.im * c.im).sqrt();
                }
            }
        }
        Some(out)
    }

    fn magnitudes_via_fft(&self, signal: &[f32], hop: usize, frames: usize) -> Option<Vec<f32>> {
        use rustfft::num_complex::Complex;

        let k = self.re.len();
        let n_sig = signal.len();
        let half = self.half as isize;
        if n_sig <= k || hop == 0 || frames == 0 { return None; }

        // Overlap-save: `step` new outputs per transform, so N wants to be
        // comfortably larger than K or most of each transform is overlap.
        let n = fft_block_len(k);
        if n <= k { return None; }
        let step = n - k + 1;

        // Frames whose whole window lies inside the signal. Outside this range
        // the direct path's edge renormalisation applies.
        let first = (half as usize).div_ceil(hop);
        let last_pos = (n_sig as isize) - 1 - half;
        if last_pos < 0 { return None; }
        let last = ((last_pos as usize) / hop).min(frames.saturating_sub(1));
        if first > last { return None; }

        let (fwd, inv) = ASLT_PLANNER.with(|p| {
            let mut p = p.borrow_mut();
            (p.plan_fft_forward(n), p.plan_fft_inverse(n))
        });

        // Time-reversed conjugate-free kernel: correlating with w is convolving
        // with w reversed, which is what the transform pair actually computes.
        let mut kernel = vec![Complex { re: 0.0f32, im: 0.0f32 }; n];
        for (slot, i) in kernel.iter_mut().zip((0..k).rev()) {
            *slot = Complex { re: self.re[i], im: self.im[i] };
        }
        fwd.process(&mut kernel);

        let mut out = vec![0.0f32; frames];
        // Edge frames, direct.
        for (fi, slot) in out.iter_mut().enumerate() {
            if fi < first || fi > last {
                *slot = self.response(signal, (fi * hop) as isize);
            }
        }

        let scale = 1.0 / n as f32;
        let mut buf = vec![Complex { re: 0.0f32, im: 0.0f32 }; n];
        let mut base = 0usize;
        while base < n_sig {
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = Complex {
                    re: signal.get(base + i).copied().unwrap_or(0.0),
                    im: 0.0,
                };
            }
            fwd.process(&mut buf);
            for (b, h) in buf.iter_mut().zip(kernel.iter()) { *b *= *h; }
            inv.process(&mut buf);

            // Linear-convolution outputs y[m] land at buf[m - base] for
            // m in [base + k - 1, base + n).
            let m_lo = base + k - 1;
            let m_hi = (base + n).min(n_sig);
            // R[t] = y[t + half], t = fi·hop.
            let fi_lo = ((m_lo as isize - half).max(0) as usize).div_ceil(hop).max(first);
            let fi_hi = if (m_hi as isize) - 1 - half < 0 {
                0
            } else {
                ((((m_hi as isize) - 1 - half) as usize) / hop).min(last)
            };
            if fi_lo <= fi_hi {
                for (fi, slot) in (fi_lo..=fi_hi).zip(out[fi_lo..=fi_hi].iter_mut()) {
                    let c = buf[fi * hop + half as usize - base];
                    *slot = 2.0 * scale * (c.re * c.re + c.im * c.im).sqrt();
                }
            }
            base += step;
        }
        Some(out)
    }
}

// ---------------------------------------------------------------------------
// Frequency-domain convolution
// ---------------------------------------------------------------------------

// A wavelet response is a correlation, and long correlations belong in the
// frequency domain. The catch is that the direct path only evaluates hop-spaced
// outputs, costing `frames × K`, while an FFT convolution computes *every*
// output at `L log N` — so the FFT only wins once the kernel is long relative to
// the hop. That is precisely the low end: at 20 Hz and grid Q the kernel runs
// ~440 000 taps against a hop of 267, and the direct path is 25× the work.
//
// Both routes are exact, so the threshold below is purely a performance choice;
// `fft_route_matches_direct` holds the two to the same answer.

/// Don't bother below this kernel length — small transforms lose to the direct
/// loop on overheads alone.
const FFT_MIN_KERNEL: usize = 2048;
/// Cap on transform size, to bound peak memory. Each worker holds two buffers of
/// this length, so 2²¹ is ~16 MB apiece and ~256 MB across eight threads.
const FFT_MAX_LOG2: u32 = 21;

/// Tap count a Morlet of these parameters will have, without building it.
///
/// Planning which block sizes a run needs means knowing every kernel's length
/// up front — and the kernels themselves are far too large to hold all at once
/// (the lowest bars run to hundreds of thousands of taps apiece), so the length
/// has to be derivable without allocating. Mirrors [`Morlet::new`].
fn morlet_taps(freq: f32, cycles: f32, sample_rate: f32) -> usize {
    let sigma = sigma_samples(freq, cycles, sample_rate).max(0.5);
    2 * ((SUPPORT_SIGMAS * sigma).ceil() as usize).max(1) + 1
}

/// Cycle counts of a superlet's members, in the order [`Superlet::new`] builds
/// them. Kept beside that constructor so the two cannot drift apart.
fn superlet_cycles(q: f32, cfg: &AsltConfig) -> Vec<f32> {
    let c_max = c_max_for_q(q, cfg);
    let n = cfg.n_wavelets.max(1);
    let a = cfg.spread.clamp(0.0, 1.0);
    (0..n).map(|i| {
        let frac = if n == 1 { 1.0 } else { a + (1.0 - a) * i as f32 / (n - 1) as f32 };
        (c_max * frac).max(DEFAULT_C_MIN)
    }).collect()
}

/// Smallest block size worth sending to the GPU.
///
/// Measured against the CPU route with all eight cores busy: 0.46× at 2^15,
/// 1.87× at 2^17, 6.84× at 2^19. The crossover is real and falls the useful
/// way — the large blocks are the low-frequency bars whose kernels run to
/// hundreds of thousands of taps, and those dominate the expensive presets.
/// Below this the CPU is simply quicker, so it keeps that work.
const GPU_MIN_BLOCK: usize = 1 << 17;

/// Live threshold. **On where a device is available; `MOOSIK_GPU=0` forces it
/// off.**
///
/// Measured on a 20 s signal, 1024 bars, per 5 minutes of audio, against the
/// same build with the device switched off:
///
/// | preset   | CPU only | GPU  |       |
/// |----------|----------|------|-------|
/// | Fast     | 0.7      | 0.7  | —     |
/// | Standard | 1.3      | 1.3  | —     |
/// | High     | 3.2      | 2.6  | 1.23× |
/// | Ultra    | 6.0      | 4.9  | 1.22× |
/// | Extreme  | 12.7     | 11.6 | 1.09× |
///
/// It took four wrong answers to get here, and the record is worth keeping,
/// because every one of them was about the device and none of them was right.
/// Routing to the GPU first made Extreme *slower* — 19.4 against 12.7 — and the
/// theories for why were pipeline syncs, phase serialisation, and shader speed.
/// Per-phase timings (`MOOSIK_GPU_TRACE=1`, which is why that flag still exists)
/// showed the device taking 0.11 s per chunk against 0.86 s for the fill that
/// followed it: roughly 10 s of device time and 48 s of CPU time in a 69 s run.
/// The device had never been the cost.
///
/// What actually fixed it, in order of how much each was worth:
///
/// * the fill parallelised across *bars*, and the kernel staging budget only
///   fits three or four bars per chunk — so the most expensive phase in the run
///   was putting three work items on an eight-core machine. Their wavelets are
///   independent, and flattening to one item per wavelet cut the fill from
///   0.86 s to 0.46 s;
/// * wavelets the device declined were computed with the bare `magnitudes`,
///   which transforms the whole signal again per wavelet, instead of the
///   group's shared blocks the CPU route has always used;
/// * a chunk's kernels — millions of taps of transcendentals — were built one
///   bar after another on a single thread while the CPU route built its own
///   inside a `par_iter`;
/// * device and CPU bars ran in sequence rather than at once.
///
/// A machine with no usable device falls back to the CPU route, which is
/// mandatory anyway and is what these numbers are measured against.
static GPU_MIN_LIVE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

fn gpu_min_block() -> usize {
    use std::sync::atomic::Ordering as O;
    let v = GPU_MIN_LIVE.load(O::Relaxed);
    if v != usize::MAX { return v; }
    let init = match std::env::var_os("MOOSIK_GPU") {
        Some(v) if v == "0" => usize::MAX - 1,
        _ => GPU_MIN_BLOCK,
    };
    GPU_MIN_LIVE.store(init, O::Relaxed);
    init
}

/// How a block size is routed.
///
/// Supplied as an argument rather than read from a mutable global. The global
/// version was writable by tests, and two of them steered each other through
/// it: one rewrote the threshold three times while another demanded bit-exact
/// equality between consecutive analyses, which routed one call to the device
/// and one to the cores. Those agree to a tolerance, not bit for bit, so the
/// second test failed intermittently under a parallel suite and passed alone.
type RouteDecision = fn(usize) -> bool;

/// Whether a group of this block size goes to the device.
///
/// `MOOSIK_GPU=0` is absolute; otherwise the answer comes from what this
/// machine has actually been measured doing, not from a constant compiled in
/// from someone else's hardware.
fn route_to_device(group_n: usize) -> bool {
    if group_n < gpu_min_block() { return false }
    #[cfg(test)]
    { return true }
    #[cfg(not(test))]
    { super::gpu_calib::use_device(group_n) }
}

/// `MOOSIK_GPU_TRACE=1` — per-chunk timings for the four phases of the device
/// route, so where its time goes is a measurement rather than a guess. Kept
/// because every structural theory about this route so far has been wrong, and
/// each one cost a benchmark run to disprove.
fn gpu_trace() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering as O};
    static ON: AtomicU8 = AtomicU8::new(u8::MAX);
    let v = ON.load(O::Relaxed);
    if v != u8::MAX { return v == 1 }
    let init = u8::from(std::env::var_os("MOOSIK_GPU_TRACE").is_some_and(|v| v == "1"));
    ON.store(init, O::Relaxed);
    init == 1
}

/// Where an analysis actually spent its time, accumulated live.
///
/// Everything here is measured, never modelled. The device route was rewritten
/// four times on reasoning that turned out to be wrong about which phase was
/// expensive, and the only thing that settled it was timing the phases — so the
/// timings are now a permanent, readable part of the run rather than something
/// bolted on when a question comes up.
///
/// Nanoseconds summed across threads, so the phase totals exceed wall-clock on a
/// parallel run; they are for comparing phases against each other, which is the
/// question they exist to answer. `device` is the exception and is close to real
/// elapsed time, because dispatches are serialised through one thread per group.
#[derive(Default)]
pub struct Stats {
    pub chunks: std::sync::atomic::AtomicU64,
    pub kernels: std::sync::atomic::AtomicU64,
    pub build_ns: std::sync::atomic::AtomicU64,
    pub stage_ns: std::sync::atomic::AtomicU64,
    pub device_ns: std::sync::atomic::AtomicU64,
    pub fill_ns: std::sync::atomic::AtomicU64,
    /// Wall-clock of the whole `analyze_with_progress` call.
    pub total_ns: std::sync::atomic::AtomicU64,
    /// Bars that went to the device, and bars that stayed on the cores.
    pub gpu_bars: std::sync::atomic::AtomicU64,
    pub cpu_bars: std::sync::atomic::AtomicU64,
}

/// Snapshot of [`Stats`], detached from the atomics so the UI can hold it.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct StatsSnapshot {
    pub chunks: u64,
    pub kernels: u64,
    pub build_s: f64,
    pub stage_s: f64,
    pub device_s: f64,
    pub fill_s: f64,
    pub total_s: f64,
    pub gpu_bars: u64,
    pub cpu_bars: u64,
}

impl StatsSnapshot {
    /// Device time as a fraction of the whole analysis — what a GPU monitor
    /// shows while a track is being processed.
    ///
    /// It reads far below 100%, and that is not waste to be eliminated: most of
    /// an analysis is bars whose kernels are too short to be worth sending
    /// anywhere near the device, and those run on the cores by design. The
    /// dispatches are now overlapped with the assembly beside them, so what is
    /// left is a genuine division of labour rather than the device waiting its
    /// turn.
    pub fn device_duty(&self) -> f64 {
        if self.total_s <= 0.0 { 0.0 } else { self.device_s / self.total_s }
    }

    pub fn used_gpu(&self) -> bool { self.chunks > 0 }
}

pub static STATS: std::sync::LazyLock<Stats> = std::sync::LazyLock::new(Stats::default);

/// Read the last run's accounting.
pub fn stats() -> StatsSnapshot {
    use std::sync::atomic::Ordering as O;
    let ns = |a: &std::sync::atomic::AtomicU64| a.load(O::Relaxed) as f64 / 1e9;
    StatsSnapshot {
        chunks: STATS.chunks.load(O::Relaxed),
        kernels: STATS.kernels.load(O::Relaxed),
        build_s: ns(&STATS.build_ns),
        stage_s: ns(&STATS.stage_ns),
        device_s: ns(&STATS.device_ns),
        fill_s: ns(&STATS.fill_ns),
        total_s: ns(&STATS.total_ns),
        gpu_bars: STATS.gpu_bars.load(O::Relaxed),
        cpu_bars: STATS.cpu_bars.load(O::Relaxed),
    }
}

/// The device the transform would use, if any — for the debug panel, so
/// "is it even on the GPU?" is answerable without reading the code.
pub fn gpu_device_name() -> Option<String> {
    if gpu_min_block() > (1usize << 31) { return None; }
    // Cached: `describe` builds a wgpu instance and requests an adapter, and
    // the debug panel asks once a frame.
    static NAME: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    NAME.get_or_init(super::gpu::GpuFft::describe).clone()
}

fn stats_reset() {
    use std::sync::atomic::Ordering as O;
    for a in [
        &STATS.chunks, &STATS.kernels, &STATS.build_ns, &STATS.stage_ns,
        &STATS.device_ns, &STATS.fill_ns, &STATS.total_ns,
        &STATS.gpu_bars, &STATS.cpu_bars,
    ] {
        a.store(0, O::Relaxed);
    }
}

/// Transforms a batch must contain before the GPU is worth using.
///
/// Size alone is the wrong test. At the largest block sizes a short signal
/// yields a single block, so a chunk is a handful of enormous transforms with
/// no width to them — the device sits mostly idle while eight CPU cores chew
/// through the same bars in parallel. Measured on a 20 s signal that turned
/// Extreme from 13.0 into 18.6 minutes; the same block size on a five-minute
/// track has fourteen blocks and is firmly worth it. Gating on `blocks ×
/// kernels` rather than on `n` is what tells those two cases apart.
const GPU_MIN_BATCH: usize = 24;

/// Taps to hold at once while staging a chunk of kernels for the GPU.
///
/// A single kernel at the largest block size is a few megabytes, and a block
/// size can cover a whole octave of bars — hundreds of kernels. They cannot all
/// be built at once, so the group is walked in chunks.
const GPU_KERNEL_TAP_BUDGET: usize = 48 << 20;

/// Weighted geometric mean of per-wavelet magnitudes — the superlet itself,
/// applied to columns that may have come from anywhere.
fn combine_logs(mags: &[Vec<f32>], weights: &[f32], total_weight: f32, frames: usize) -> Vec<f32> {
    let mut acc = vec![0.0f32; frames];
    for (col, w) in mags.iter().zip(weights) {
        for (a, m) in acc.iter_mut().zip(col) {
            *a += w * m.max(MAG_FLOOR).ln();
        }
    }
    let inv = 1.0 / total_weight.max(f32::EPSILON);
    acc.iter().map(|a| (a * inv).exp()).collect()
}

/// Ceiling on one shared signal transform, in bytes.
///
/// Sharing trades memory for time: the blocks for a size stay resident while
/// every kernel of that size uses them. At the largest transform this is a few
/// hundred megabytes, which is worth it — but it must not be allowed to grow
/// without bound on a long track, so a size that would exceed this simply falls
/// back to the per-kernel route.
const SHARED_BLOCKS_MAX_BYTES: usize = 512 << 20;

/// The signal, forward-transformed once per block size and shared by every
/// kernel that uses that size.
///
/// Overlap-save transforms the signal block by block for each kernel, and the
/// signal does not change between kernels — on a five-minute track at Extreme
/// that is ~5.6 million forward transforms where 14 thousand would do. They
/// could not be shared before because the block stride was `n - k + 1`, which
/// depends on the kernel; fixing it at `n/2` makes the decomposition identical
/// for every kernel of a size, and stays legal because `n >= 2k` always, so
/// `n/2 <= n - k + 1`.
///
/// The trade is real: a shorter stride means more blocks, so the *inverse*
/// transforms — which cannot be shared — go up. Measured end to end that is
/// still a 1.59× win, not the 580× the forward saving alone suggests.
pub struct SignalBlocks {
    /// Transform length.
    pub n: usize,
    /// Distance between block starts.
    pub stride: usize,
    /// `blocks × n` forward transforms, concatenated.
    data: Vec<rustfft::num_complex::Complex<f32>>,
}

impl SignalBlocks {
    pub fn blocks(&self) -> usize { self.data.len() / self.n.max(1) }

    /// Forward-transform `signal` in `n`-sized blocks stepping by `n/2`.
    pub fn build(signal: &[f32], n: usize) -> Self {
        use rustfft::num_complex::Complex;
        let stride = (n / 2).max(1);
        let blocks = signal.len().div_ceil(stride).max(1);
        let mut data = vec![Complex { re: 0.0f32, im: 0.0 }; blocks * n];
        let fwd = ASLT_PLANNER.with(|p| p.borrow_mut().plan_fft_forward(n));
        for b in 0..blocks {
            let base = b * stride;
            let dst = &mut data[b * n..(b + 1) * n];
            for (i, slot) in dst.iter_mut().enumerate() {
                *slot = Complex { re: signal.get(base + i).copied().unwrap_or(0.0), im: 0.0 };
            }
            fwd.process(dst);
        }
        Self { n, stride, data }
    }

    fn block(&self, b: usize) -> &[rustfft::num_complex::Complex<f32>] {
        &self.data[b * self.n..(b + 1) * self.n]
    }
}

/// Transform length overlap-save would use for a kernel of `k` taps.
pub fn fft_block_len(k: usize) -> usize {
    (2 * k).next_power_of_two().min(1usize << FFT_MAX_LOG2)
}

/// Empirical correction to the direct path's modelled cost.
///
/// Counting arithmetic alone says break-even sits near `k ≈ 4·hop·log₂N`, about
/// 16 000 taps at a 267-sample hop. Measured, that threshold is far too strict:
/// routing everything above ~2 000 taps through the transform instead made the
/// Fast preset more than twice as quick. The model undercounts because the
/// direct path re-reads a long slice of the signal for every output and is
/// bandwidth-bound, while `rustfft` runs close to peak on cache-resident blocks.
/// 8 is that gap, from the benchmark rather than from theory.
const FFT_DIRECT_COST_FACTOR: f64 = 8.0;

/// Is the frequency-domain route actually cheaper for this kernel and hop?
///
/// Work per *input sample*: the direct path evaluates one output every `hop`
/// samples at `k` taps each, so `k/hop`; overlap-save spends `2·N·log₂N` per
/// `step = N−k+1` samples, independent of hop. Both routes are exact, so getting
/// this wrong costs time and nothing else.
fn fft_is_cheaper(k: usize, hop: usize) -> bool {
    if k < FFT_MIN_KERNEL || hop == 0 { return false; }
    let n = fft_block_len(k);
    if n <= k { return false; }
    let step = (n - k + 1) as f64;
    let log_n = n.trailing_zeros().max(1) as f64;
    let direct = (k as f64 / hop as f64) * FFT_DIRECT_COST_FACTOR;
    let transformed = 2.0 * n as f64 * log_n / step;
    direct > transformed * 1.3
}

thread_local! {
    /// Planner per worker: `rustfft` caches plans inside an instance, and one bar
    /// reuses the same handful of sizes across all of its wavelets.
    static ASLT_PLANNER: std::cell::RefCell<rustfft::FftPlanner<f32>> =
        std::cell::RefCell::new(rustfft::FftPlanner::new());
}

// ---------------------------------------------------------------------------
// Superlet
// ---------------------------------------------------------------------------

/// The wavelet set for one bar, built once and reused across every frame.
struct Superlet {
    wavelets: Vec<Morlet>,
    /// Per-wavelet weight in the geometric mean. All 1.0 except the last, which
    /// carries the fractional part of the order.
    weights: Vec<f32>,
    /// Sum of `weights` — the fractional order itself.
    total_weight: f32,
}

impl Superlet {
    /// Build the set that hits `q` at `freq`.
    ///
    /// Cycle counts are derived from the Q target rather than chosen: the
    /// longest member is whatever reaches `q`, and the rest fill in evenly down
    /// to `spread × c_max`. Members carry equal weight — the fractional-order
    /// weighting the paper uses existed to stop integer order steps showing as
    /// seams across the spectrum, and with Q varying continuously there are no
    /// steps left to hide.
    fn new(freq: f32, q: f32, cfg: &AsltConfig, sample_rate: f32) -> Self {
        let c_max = c_max_for_q(q, cfg);
        let n = cfg.n_wavelets.max(1);
        let a = cfg.spread.clamp(0.0, 1.0);

        let mut wavelets = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for i in 0..n {
            let frac = if n == 1 { 1.0 } else { a + (1.0 - a) * i as f32 / (n - 1) as f32 };
            wavelets.push(Morlet::new(freq, (c_max * frac).max(DEFAULT_C_MIN), sample_rate));
            weights.push(1.0);
        }

        let total_weight = weights.iter().sum::<f32>().max(f32::EPSILON);
        Self { wavelets, weights, total_weight }
    }

    /// Weighted geometric mean of the member responses, in log space.
    fn response(&self, signal: &[f32], centre: isize) -> f32 {
        let mut acc = 0.0f32;
        for (w, weight) in self.wavelets.iter().zip(self.weights.iter()) {
            let mag = w.response(signal, centre).max(MAG_FLOOR);
            acc += weight * mag.ln();
        }
        (acc / self.total_weight).exp()
    }

    /// Whether any member is long enough to be worth transforming.
    ///
    /// Decided per superlet rather than per wavelet so the whole bar takes one
    /// route: the members within a set differ only by `spread`, so they cross
    /// the threshold together, and mixing routes inside a bar would only
    /// complicate cancellation.
    fn prefers_fft(&self, hop: usize) -> bool {
        self.wavelets.iter().any(|w| fft_is_cheaper(w.re.len(), hop))
    }

    /// Every frame at once, so each member can take the frequency-domain route.
    ///
    /// The per-frame [`Superlet::response`] cannot: an FFT convolution only pays
    /// off across a whole signal, and interleaving members per frame would
    /// rebuild the same transform for every one.
    fn responses(&self, signal: &[f32], hop: usize, frames: usize) -> Vec<f32> {
        self.responses_with(signal, hop, frames, &Default::default())
    }

    /// As [`Superlet::responses`], but taking whatever shared signal transforms
    /// are available.
    ///
    /// A member whose block size is not in `shared` simply falls back to its own
    /// forward pass, so this is always correct and only ever faster — which
    /// matters because the shared set is deliberately bounded by memory and can
    /// legitimately be missing a size.
    fn responses_with(
        &self, signal: &[f32], hop: usize, frames: usize,
        shared: &std::collections::HashMap<usize, std::sync::Arc<SignalBlocks>>,
    ) -> Vec<f32> {
        let mut acc = vec![0.0f32; frames];
        for (w, weight) in self.wavelets.iter().zip(self.weights.iter()) {
            let k = w.re.len();
            let mags = shared
                .get(&fft_block_len(k))
                .filter(|_| fft_is_cheaper(k, hop))
                .and_then(|b| w.magnitudes_via_shared(signal, b, hop, frames))
                .unwrap_or_else(|| w.magnitudes(signal, hop, frames));
            for (a, m) in acc.iter_mut().zip(mags) {
                *a += weight * m.max(MAG_FLOOR).ln();
            }
        }
        let inv = 1.0 / self.total_weight;
        acc.iter().map(|a| (a * inv).exp()).collect()
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Number of frames `analyze` will produce for a signal of `len` samples.
pub fn frame_count(len: usize, hop: usize) -> usize {
    if len == 0 || hop == 0 { 0 } else { len.div_euclid(hop).max(1) }
}

/// Hop, in samples, for a target frame rate. Frame rate is defined in *time*,
/// not in samples, so cost stays linear in sample rate instead of quadratic —
/// which is what keeps 176.4 kHz DSD analysis from costing 16× a 44.1 kHz track.
pub fn hop_for_fps(sample_rate: u32, fps: f32) -> usize {
    ((sample_rate as f32 / fps.max(1.0)).round() as usize).max(1)
}

/// Full ASLT spectrogram: `frames × n_bars` of raw magnitudes.
///
/// Magnitudes are linear and un-weighted — 1.0 means "a unit-amplitude sine sits
/// exactly on this bar". dB conversion and ISO 226 weighting belong to the
/// caller.
///
/// Parallelised over *bars*, not frames: every frame at a given bar reuses the
/// same wavelet tables, so this builds each set once. Going per-frame instead
/// would either rebuild them 54 000 times or hold gigabytes of tables at once.
#[allow(clippy::too_many_arguments)]
pub fn analyze(
    signal: &[f32],
    sample_rate: u32,
    n_bars: usize,
    min_freq: f32,
    max_freq: f32,
    hop: usize,
    cfg: &AsltConfig,
) -> Vec<Vec<f32>> {
    analyze_with_progress(
        signal, sample_rate, n_bars, min_freq, max_freq, hop, cfg,
        &|| true, &|_| {},
    )
}

/// As [`analyze`], but cancellable and instrumented.
///
/// `should_continue` is polled *before* work is committed to, both per bar and
/// periodically inside the long low-frequency bars; returning `false` abandons
/// the run and yields an empty result. `on_bar_done` fires once per finished bar
/// and exists purely for progress reporting.
///
/// Two callbacks rather than one because the single combined version had to be
/// invoked after a bar's work to report it, which meant "abort" could not
/// prevent any work at all — every remaining bar still computed in full and
/// cancelling a 20-minute run took 20 minutes.
#[allow(clippy::too_many_arguments)]
pub fn analyze_with_progress(
    signal: &[f32],
    sample_rate: u32,
    n_bars: usize,
    min_freq: f32,
    max_freq: f32,
    hop: usize,
    cfg: &AsltConfig,
    should_continue: &(dyn Fn() -> bool + Sync),
    on_bar_done: &(dyn Fn(usize) + Sync),
) -> Vec<Vec<f32>> {
    analyze_routed(
        signal, sample_rate, n_bars, min_freq, max_freq, hop, cfg,
        should_continue, on_bar_done, route_to_device,
    )
}

/// [`analyze_with_progress`], with the device-routing decision supplied.
///
/// Production passes [`route_to_device`]; tests pass whatever they are actually
/// asserting about, so that no two tests can steer each other by writing to a
/// shared threshold.
#[allow(clippy::too_many_arguments)]
fn analyze_routed(
    signal: &[f32],
    sample_rate: u32,
    n_bars: usize,
    min_freq: f32,
    max_freq: f32,
    hop: usize,
    cfg: &AsltConfig,
    should_continue: &(dyn Fn() -> bool + Sync),
    on_bar_done: &(dyn Fn(usize) + Sync),
    route: RouteDecision,
) -> Vec<Vec<f32>> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    let frames = frame_count(signal.len(), hop);
    if frames == 0 || n_bars == 0 { return Vec::new(); }
    stats_reset();
    let t_run = std::time::Instant::now();

    let sr = sample_rate as f32;
    let nyquist = sr * 0.5;
    // Per bar, not one number for the whole spectrum. Under `Log` this returns
    // the same closed form it always did, identically for every bar; under
    // `Erb` the bars are not evenly spaced in log frequency, so the resolution
    // the grid can draw varies along it.
    let grid_at = |bar: usize| cfg.scale.grid_q_at(bar, n_bars, min_freq, max_freq);
    let cancelled = AtomicBool::new(false);

    // How often to re-check cancellation inside one bar. The lowest bars run for
    // tens of seconds on their own, so a per-bar check alone would leave Abort
    // feeling ignored.
    const CANCEL_CHECK_FRAMES: usize = 512;
    // Cancellation granularity on the frequency-domain route is one whole bar,
    // since a transform cannot be stopped part-way. Those bars are the fast ones
    // now, so the wait is short — but the cheap direct route keeps its
    // finer-grained checks, because there a single bar really can run for
    // tens of seconds.
    let cancel_at_bar = |bar: usize, cancelled: &AtomicBool| -> bool {
        if cancelled.load(Ordering::Relaxed) || !should_continue() {
            cancelled.store(true, Ordering::Relaxed);
            return true;
        }
        let _ = bar;
        false
    };

    // Plan every bar before computing any of it. Kernel *lengths* follow from
    // the parameters, so which block sizes the run needs can be worked out
    // without building a single kernel — which matters, because the lowest bars
    // run to hundreds of thousands of taps and could never all be held at once.
    //
    // Q comes from the grid alone. Letting per-band content lower it —
    // shortening the window where the pitch is sweeping — was tried and
    // removed: a band whose Q differs from its neighbours' reads at a different
    // level on anything broadband (+9.5 dB measured), because a wavelet is
    // normalised so a *tone* reads 1.0 whatever its bandwidth, while noise reads
    // proportional to √bandwidth.
    struct BarPlan { bar: usize, freq: f32, q: f32, sizes: Vec<usize>, max_n: usize }
    let mut plans: Vec<BarPlan> = Vec::new();
    let mut direct_bars: Vec<(usize, f32, f32)> = Vec::new();
    let mut above_nyquist: Vec<usize> = Vec::new();

    for bar in 0..n_bars {
        let f = cfg.scale.bar_center(bar, n_bars, min_freq, max_freq);
        // Above Nyquist there is nothing to measure; the FFT path clamps by bin
        // index, but a wavelet would happily alias instead.
        if f >= nyquist { above_nyquist.push(bar); continue; }
        let q = effective_q_at(f, grid_at(bar), cfg);
        let mut sizes: Vec<usize> = superlet_cycles(q, cfg).into_iter()
            .map(|c| morlet_taps(f, c, sr))
            .filter(|&k| fft_is_cheaper(k, hop))
            .map(fft_block_len)
            .filter(|&n| n > 0)
            .collect();
        sizes.sort_unstable();
        sizes.dedup();
        match sizes.last().copied() {
            Some(max_n) => plans.push(BarPlan { bar, freq: f, q, sizes, max_n }),
            None => direct_bars.push((bar, f, q)),
        }
    }

    let mut columns: Vec<Option<Vec<f32>>> = vec![None; n_bars];
    for bar in above_nyquist {
        columns[bar] = Some(vec![0.0; frames]);
        on_bar_done(bar);
    }

    // Bars sharing a longest kernel are computed together, so the signal
    // transform for that size is built once and used by all of them. Largest
    // first: those are the expensive bars, and if the memory budget only
    // stretches to a few sizes they are the ones worth spending it on.
    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (i, p) in plans.iter().enumerate() {
        groups.entry(p.max_n).or_default().push(i);
    }

    let block_bytes = |_n: usize| {
        // A block set is always ~2 signal-lengths of complex samples, whatever
        // the transform size: bigger blocks mean proportionally fewer of them.
        2 * signal.len() * std::mem::size_of::<rustfft::num_complex::Complex<f32>>()
    };

    for (_key, idxs) in groups.iter().rev() {
        if cancelled.load(Ordering::Relaxed) { break; }

        let mut wanted: Vec<usize> = idxs.iter()
            .flat_map(|&i| plans[i].sizes.iter().copied())
            .collect();
        wanted.sort_unstable_by(|a, b| b.cmp(a)); // largest first
        wanted.dedup();

        let mut shared: std::collections::HashMap<usize, std::sync::Arc<SignalBlocks>> =
            Default::default();
        let mut spent = 0usize;
        for n in wanted {
            let cost = block_bytes(n);
            if spent + cost > SHARED_BLOCKS_MAX_BYTES { break; }
            spent += cost;
            shared.insert(n, std::sync::Arc::new(SignalBlocks::build(signal, n)));
        }

        let members: std::collections::HashSet<usize> =
            idxs.iter().map(|&i| plans[i].bar).collect();
        let by_bar: std::collections::HashMap<usize, &BarPlan> =
            idxs.iter().map(|&i| (plans[i].bar, &plans[i])).collect();

        // Big blocks go to the GPU when there is one. The convolution is the
        // same overlap-save either way — this only changes where the multiply
        // and the inverse happen, and the result is held to the CPU's by
        // `convolve_matches_the_cpu_route`.
        let group_n = *_key;
        let stride = (group_n / 2).max(1);
        let blocks = signal.len().div_ceil(stride).max(1);
        let gpu = route(group_n)
            .then(super::gpu::GpuFft::shared)
            .flatten();
        let t_group = std::time::Instant::now();

        let bars: Vec<usize> = idxs.iter().map(|&i| plans[i].bar).collect();

        // Split the group's bars between device and cores *before* either side
        // starts, so the two can run at the same time.
        //
        // This is the difference between the GPU being a net win and a net
        // loss. Measured end to end, running every GPU bar to completion and
        // only then starting the CPU pass was slower than not using the device
        // at all (Extreme: 19.4 min against 13.0 CPU-only) — eight cores idled
        // through the device phase and the device idled through theirs, so the
        // total was the sum of two phases instead of the longer of two. Making
        // the shader faster could not have fixed that.
        //
        // The split is free to make up front because eligibility follows from
        // kernel *lengths*, which are a function of the parameters: not one
        // wavelet has to be built to know who gets what.
        let mut chunks: Vec<(usize, usize)> = Vec::new();
        let mut gpu_bars: std::collections::HashSet<usize> = Default::default();
        if gpu.is_some() {
            let mut at = 0usize;
            while at < bars.len() {
                // A chunk of bars whose kernels fit the staging budget. The
                // lowest bars carry millions of taps apiece, so this is not a
                // formality.
                let mut taps = 0usize;
                let mut end = at;
                while end < bars.len() {
                    let p = by_bar[&bars[end]];
                    let cost: usize = superlet_cycles(p.q, cfg).into_iter()
                        .map(|c| morlet_taps(p.freq, c, sr) * 3 * 4)
                        .sum();
                    if end > at && taps + cost > GPU_KERNEL_TAP_BUDGET { break; }
                    taps += cost;
                    end += 1;
                }

                // Too little width to fill the device: those bars are quicker
                // on the cores, which run them across all of them at once.
                let eligible: usize = bars[at..end].iter().map(|&b| {
                    let p = by_bar[&b];
                    superlet_cycles(p.q, cfg).into_iter()
                        .map(|c| morlet_taps(p.freq, c, sr))
                        .filter(|&k| fft_is_cheaper(k, hop) && fft_block_len(k) == group_n)
                        .count()
                }).sum();
                if blocks * eligible >= GPU_MIN_BATCH {
                    chunks.push((at, end));
                    gpu_bars.extend(bars[at..end].iter().copied());
                }
                at = end;
            }
        }

        let cpu_bars: Vec<usize> =
            bars.iter().copied().filter(|b| !gpu_bars.contains(b)).collect();
        {
            use std::sync::atomic::Ordering as O;
            STATS.gpu_bars.fetch_add(gpu_bars.len() as u64, O::Relaxed);
            STATS.cpu_bars.fetch_add(cpu_bars.len() as u64, O::Relaxed);
        }

        // A chunk whose transforms are done and whose columns still have to be
        // assembled. Held over one iteration so the assembly can run while the
        // device is busy with the chunk after it.
        struct Staged {
            sls: Vec<(usize, Superlet)>,
            owner: Vec<(usize, usize)>,
            cols: Vec<Vec<f32>>,
        }

        // Turn a finished chunk into finished bars. Pulled out of the loop
        // because it now runs one iteration behind the dispatch that produced
        // it, and once more at the end to drain the last one.
        let assemble = |p: Staged| -> Vec<(usize, Vec<f32>)> {
            let t_fill = std::time::Instant::now();
            let Staged { sls, owner, cols } = p;
            let mut got: Vec<Vec<Option<Vec<f32>>>> = sls.iter()
                .map(|(_, sl)| vec![None; sl.wavelets.len()])
                .collect();
            for ((ci, wi), col) in owner.iter().zip(cols) {
                got[*ci][*wi] = Some(col);
            }
            // One work item per *wavelet*, not per bar.
            //
            // The staging budget only fits three or four of these bars at a
            // time, so parallelising across bars put three items on an
            // eight-core machine and left five idle for the whole of the most
            // expensive phase in the run. Their wavelets are independent, and
            // there are six or seven per bar, so flattening gives the pool
            // enough to fill itself. Indexed parallel iterators collect in
            // order, which the regrouping below relies on.
            let pairs: Vec<(usize, usize)> = sls.iter().enumerate()
                .flat_map(|(ci, (_, sl))| (0..sl.wavelets.len()).map(move |wi| (ci, wi)))
                .collect();
            let mags: Vec<Vec<f32>> = pairs.par_iter()
                .map(|&(ci, wi)| {
                    let w = &sls[ci].1.wavelets[wi];
                    match &got[ci][wi] {
                        // Frames the GPU marked -1 are edge frames, whose window
                        // runs past the signal and renormalises over the taps
                        // that survive — a convolution cannot express that, so
                        // they come from the direct path exactly as on the CPU
                        // route.
                        Some(col) => col.iter().enumerate()
                            .map(|(fi, &v)| if v < 0.0 {
                                w.response(signal, (fi * hop) as isize)
                            } else { v })
                            .collect(),
                        // The wavelets the device did not take are the smaller
                        // members of the same bar, and they get the group's
                        // shared signal transform exactly as they would on the
                        // CPU route — calling the bare `magnitudes` here instead
                        // cost a full-signal forward FFT per wavelet.
                        None => {
                            let k = w.taps();
                            shared
                                .get(&fft_block_len(k))
                                .filter(|_| fft_is_cheaper(k, hop))
                                .and_then(|b| w.magnitudes_via_shared(signal, b, hop, frames))
                                .unwrap_or_else(|| w.magnitudes(signal, hop, frames))
                        }
                    }
                })
                .collect();

            let mut out = Vec::with_capacity(sls.len());
            let mut taken = 0usize;
            for (bar, sl) in sls.iter() {
                let n = sl.wavelets.len();
                let col = combine_logs(
                    &mags[taken..taken + n], &sl.weights, sl.total_weight, frames,
                );
                taken += n;
                on_bar_done(*bar);
                out.push((*bar, col));
            }
            STATS.fill_ns.fetch_add(
                t_fill.elapsed().as_nanos() as u64, Ordering::Relaxed);
            out
        };

        let (gpu_cols, cpu_cols) = rayon::join(
            || -> Vec<(usize, Vec<f32>)> {
                let Some(g) = gpu else { return Vec::new() };
                let mut out: Vec<(usize, Vec<f32>)> = Vec::new();
                // Once for the whole block size, never per chunk — and not at
                // all until a chunk is actually going to use it. Preparing
                // eagerly cost an upload and a full-signal transform for every
                // large group whose chunks then turned out to be too small to
                // be worth the device, which was most of Extreme's.
                let mut prepared: Option<super::gpu::GpuSignal> = None;
                // The chunk whose transforms are finished but whose columns are
                // not yet assembled — always exactly one behind the dispatch.
                let mut staged: Option<Staged> = None;

                for &(at, end) in &chunks {
                    if cancelled.load(Ordering::Relaxed) { break; }
                    if !should_continue() {
                        cancelled.store(true, Ordering::Relaxed);
                        break;
                    }

                    let t_build = std::time::Instant::now();
                    // Across the cores, not on this one. A chunk's kernels run
                    // to millions of taps of sin/cos apiece, and building them
                    // one bar after another here was the GPU route's real cost:
                    // the CPU route has always built its kernels inside a
                    // `par_iter`, so the device was being charged for work the
                    // cores were doing in parallel all along. (Indexed parallel
                    // iterators collect in order, which `owner` below relies
                    // on.)
                    let sls: Vec<(usize, Superlet)> = bars[at..end].par_iter()
                        .map(|&b| {
                            let p = by_bar[&b];
                            (b, Superlet::new(p.freq, p.q, cfg, sr))
                        })
                        .collect();

                    // Only the members that actually use this block size; the rest
                    // of a bar's wavelets are smaller and stay on the CPU.
                    let mut kernels: Vec<super::gpu::GpuKernel> = Vec::new();
                    let mut owner: Vec<(usize, usize)> = Vec::new(); // (chunk idx, wavelet idx)
                    for (ci, (_, sl)) in sls.iter().enumerate() {
                        for (wi, w) in sl.wavelets.iter().enumerate() {
                            if fft_block_len(w.taps()) == group_n && fft_is_cheaper(w.taps(), hop) {
                                kernels.push(super::gpu::GpuKernel {
                                    re: w.re_taps(), im: w.im_taps(), half: w.half_width(),
                                });
                                owner.push((ci, wi));
                            }
                        }
                    }

                    // The split above counted these same kernels from their lengths
                    // alone. If the two ever disagreed, the bars would go unfilled
                    // rather than wrong — the sweep after the join picks them up.
                    debug_assert!(blocks * kernels.len() >= GPU_MIN_BATCH);
                    let build = t_build.elapsed();

                    let t_stage = std::time::Instant::now();
                    if prepared.is_none() {
                        prepared = g.prepare_signal(signal, group_n, stride).ok();
                    }
                    let stage = t_stage.elapsed();

                    // Dispatch this chunk and assemble the previous one at
                    // the same time.
                    //
                    // Serially a chunk was build -> dispatch -> wait -> fill,
                    // and the trace said 0.11 s of device against 0.46 s of
                    // fill: the card finished and then sat idle four times as
                    // long while the cores caught up. That idling is the 20-30%
                    // utilisation a GPU monitor shows during an analysis.
                    // Nothing here makes the device faster — it just stops it
                    // waiting for work it could already have been given.
                    // Timed *inside* the closure, not around the join. The join
                    // does not return until both halves are done, so timing it
                    // from outside would charge the device with however long the
                    // fill beside it took — and then report the result as
                    // "device time", which is the one number this whole exercise
                    // depends on being honest.
                    let (dispatch, done_prev) = rayon::join(
                        || {
                            let t = std::time::Instant::now();
                            let r = match &prepared {
                                Some(p) => {
                                    g.convolve_with(p, signal.len(), hop, frames, &kernels)
                                }
                                None => Err("signal could not be staged".into()),
                            };
                            (r, t.elapsed())
                        },
                        || staged.take().map(&assemble),
                    );
                    let (convolved, device) = dispatch;
                    if let Some(cols) = done_prev { out.extend(cols); }

                    // The kernels are slices into `sls`'s wavelets, so they have
                    // to go before it can be handed on. Nothing needs them once
                    // the dispatch has returned.
                    let n_kernels = kernels.len();
                    drop(kernels);

                    match convolved {
                        // Held over rather than assembled now: assembling here
                        // would put the device straight back to waiting, which
                        // is the thing this is for.
                        Ok(cols) if cols.len() == n_kernels => {
                            staged = Some(Staged { sls, owner, cols });
                        }
                        // A GPU that declines — out of memory, a lost device — is
                        // not a failure. Those bars simply fall through to the
                        // sweep after the join, which is the same computation.
                        _ => {}
                    }
                    {
                        use std::sync::atomic::Ordering as O;
                        let add = |a: &std::sync::atomic::AtomicU64, d: std::time::Duration| {
                            a.fetch_add(d.as_nanos() as u64, O::Relaxed);
                        };
                        STATS.chunks.fetch_add(1, O::Relaxed);
                        STATS.kernels.fetch_add(n_kernels as u64, O::Relaxed);
                        add(&STATS.build_ns, build);
                        add(&STATS.stage_ns, stage);
                        add(&STATS.device_ns, device);
                    }
                    if gpu_trace() {
                        crate::mlog!(
                            "[gpu] n={group_n:<9} bars={:<4} kernels={:<5} blocks={blocks:<3} \
                             build={:>7.2}s stage={:>6.2}s device={:>7.2}s (fill overlapped)",
                            end - at, n_kernels,
                            build.as_secs_f64(), stage.as_secs_f64(), device.as_secs_f64(),
                        );
                    }
                }
                // The final dispatch has nothing after it to overlap with.
                if let Some(p) = staged.take() { out.extend(assemble(p)); }
                out
            },
            || -> Vec<(usize, Vec<f32>)> {
                // Everything the device did not take, across every core, while
                // it works. These are the smaller-kernel bars of the group, so
                // there are usually many more of them than there are GPU bars.
                cpu_bars.par_iter()
                    .filter_map(|&bar| {
                        if cancel_at_bar(bar, &cancelled) { return None; }
                        let p = by_bar[&bar];
                        let sl = Superlet::new(p.freq, p.q, cfg, sr);
                        let col = sl.responses_with(signal, hop, frames, &shared);
                        on_bar_done(bar);
                        Some((bar, col))
                    })
                    .collect()
            },
        );

        for (bar, col) in gpu_cols.into_iter().chain(cpu_cols) {
            columns[bar] = Some(col);
        }

        // Normally empty: only a device that declined mid-run leaves anything
        // here, and then this is the identical computation on the cores.
        columns.par_iter_mut().enumerate()
            .filter(|(bar, slot)| members.contains(bar) && slot.is_none())
            .for_each(|(bar, slot)| {
                if cancel_at_bar(bar, &cancelled) { return; }
                let p = by_bar[&bar];
                let sl = Superlet::new(p.freq, p.q, cfg, sr);
                *slot = Some(sl.responses_with(signal, hop, frames, &shared));
                on_bar_done(bar);
            });

        // What this group cost, and which route paid it. Units are bars ×
        // frames so a 20 s track and a 5 minute one are comparable. A cancelled
        // group is not a measurement of anything and is dropped.
        if !cancelled.load(Ordering::Relaxed) {
            super::gpu_calib::record(
                group_n,
                (bars.len() * frames) as f64,
                t_group.elapsed().as_secs_f64(),
                gpu.is_some(),
            );
        }
    }

    // The short-kernel bars, which never touch a transform. They keep the
    // finer-grained cancellation checks: one of these really can run for tens of
    // seconds, where a frequency-domain bar cannot be stopped part-way anyway.
    if !cancelled.load(Ordering::Relaxed) {
        let direct: std::collections::HashMap<usize, (f32, f32)> =
            direct_bars.iter().map(|&(b, f, q)| (b, (f, q))).collect();
        columns.par_iter_mut().enumerate()
            .filter(|(bar, _)| direct.contains_key(bar))
            .for_each(|(bar, slot)| {
                if cancelled.load(Ordering::Relaxed) { return; }
                let (f, q) = direct[&bar];
                let sl = Superlet::new(f, q, cfg, sr);
                let mut col = Vec::with_capacity(frames);
                for fi in 0..frames {
                    if fi % CANCEL_CHECK_FRAMES == 0
                        && (cancelled.load(Ordering::Relaxed) || !should_continue())
                    {
                        cancelled.store(true, Ordering::Relaxed);
                        return;
                    }
                    col.push(sl.response(signal, (fi * hop) as isize));
                }
                *slot = Some(col);
                on_bar_done(bar);
            });
    }

    // Recorded before the cancellation check so an aborted run still reports
    // what it managed — that is exactly when someone is watching the panel.
    STATS.total_ns.store(t_run.elapsed().as_nanos() as u64, Ordering::Relaxed);

    if cancelled.load(Ordering::Relaxed) || columns.iter().any(|c| c.is_none()) {
        return Vec::new();
    }

    // Transpose to frames × bars, the layout the cache and renderer expect.
    let mut out = vec![vec![0.0f32; n_bars]; frames];
    for (bar, col) in columns.iter().enumerate() {
        let col = col.as_ref().expect("checked above");
        for (fi, &v) in col.iter().enumerate() {
            out[fi][bar] = v;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

/// Rough single-core-equivalent throughput, used only for the very first ETA
/// before any real bars have finished. Measured at ~7–8 G taps/s across 8 cores
/// on a release build; deliberately pessimistic so the estimate falls rather
/// than climbs. Replaced by the observed rate within the first few percent.
pub const TAPS_PER_SEC_HINT: f64 = 5.0e9;

/// Wavelet taps each bar costs *per frame*.
///
/// Exposed as a per-bar vector rather than a single total because bar cost spans
/// four orders of magnitude — a 20 Hz bar can be 1000× a 20 kHz one. Counting
/// completed *bars* would make the progress bar crawl and then leap; weighting
/// by taps makes it linear, and makes the ETA honest.
pub fn bar_taps_per_frame(
    sample_rate: u32,
    n_bars: usize,
    min_freq: f32,
    max_freq: f32,
    cfg: &AsltConfig,
) -> Vec<f64> {
    let sr = sample_rate as f32;
    let nyquist = sr * 0.5;
    let grid_at = |bar: usize| cfg.scale.grid_q_at(bar, n_bars, min_freq, max_freq);
    let n = cfg.n_wavelets.max(1);
    let a = cfg.spread.clamp(0.0, 1.0);
    (0..n_bars)
        .map(|bar| {
            let f = cfg.scale.bar_center(bar, n_bars, min_freq, max_freq);
            if f >= nyquist { return 0.0; }
            let c_max = c_max_for_q(effective_q_at(f, grid_at(bar), cfg), cfg);
            (0..n)
                .map(|i| {
                    let frac = if n == 1 { 1.0 } else { a + (1.0 - a) * i as f32 / (n - 1) as f32 };
                    let sigma = sigma_samples(f, (c_max * frac).max(DEFAULT_C_MIN), sr).max(0.5);
                    (2.0 * (SUPPORT_SIGMAS * sigma).ceil() + 1.0) as f64
                })
                .sum()
        })
        .collect()
}

/// Total wavelet taps `analyze` will touch — the unit of work that dominates
/// runtime. Multiply by a measured taps-per-second to get an ETA.
///
/// Closed form, so the progress dialog can quote a time before starting rather
/// than guessing from the first few percent.
pub fn estimated_taps(
    signal_len: usize,
    sample_rate: u32,
    n_bars: usize,
    min_freq: f32,
    max_freq: f32,
    hop: usize,
    cfg: &AsltConfig,
) -> f64 {
    let frames = frame_count(signal_len, hop) as f64;
    let per_frame: f64 = bar_taps_per_frame(sample_rate, n_bars, min_freq, max_freq, cfg)
        .iter()
        .sum();
    per_frame * frames
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;
    /// What `auto_fft_size_for` picks at 48 kHz: a 170.7 ms window.
    const FFT_N: usize = 8192;
    /// Signal-processing tests run at a lower rate. Cost is linear in sample
    /// rate and none of these need bandwidth above a few kHz; at 48 kHz the
    /// suite took over two minutes in a debug build.
    const SR_A: u32 = 16_000;
    /// Bass tests run lower still — the wavelets there are seconds long.
    const SR_B: u32 = 8_000;

    /// Exact centre of the bar nearest `freq`, so a test tone can be placed on a
    /// bar rather than between two. At grid-matched Q a tone landing half a bar
    /// off reads exp(-1/8) = 0.88 of full scale, which is correct behaviour and
    /// would otherwise look like a normalisation error.
    fn nearest_bar_freq(freq: f32, n_bars: usize, lo: f32, hi: f32) -> f32 {
        let bar = (0..n_bars)
            .min_by(|&a, &b| {
                let da = (bar_center_freq(a, n_bars, lo, hi) - freq).abs();
                let db = (bar_center_freq(b, n_bars, lo, hi) - freq).abs();
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        bar_center_freq(bar, n_bars, lo, hi)
    }

    fn cfg(q_ratio: f32, window: f32, n: usize, spread: f32) -> AsltConfig {
        // Adaptation off by default in tests: it depends on signal content, and
        // the parameterisation tests are about the parameters.
        AsltConfig { q_ratio, max_window_s: window, n_wavelets: n, spread, scale: Default::default() }
    }

    fn tone(freqs: &[f32], secs: f32, sr: u32) -> Vec<f32> {
        let n = (secs * sr as f32) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                freqs.iter().map(|f| (std::f32::consts::TAU * f * t).sin()).sum::<f32>()
                    / freqs.len() as f32
            })
            .collect()
    }

    fn mag_at(frames: &[Vec<f32>], freq: f32, n_bars: usize, lo: f32, hi: f32) -> f32 {
        let bar = (0..n_bars)
            .min_by(|&a, &b| {
                let da = (bar_center_freq(a, n_bars, lo, hi) - freq).abs();
                let db = (bar_center_freq(b, n_bars, lo, hi) - freq).abs();
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        frames[frames.len() / 2][bar]
    }

    /// Full width at half maximum, in Hz, with the crossings interpolated.
    ///
    /// Stepping bar-by-bar until the value drops below half overshoots by up to
    /// one bar on each side, which at grid-matched Q is most of the peak — it
    /// reported 21.6 Hz for a 12.8 Hz peak. Interpolating between the straddling
    /// bars removes the bias.
    fn half_max_width_hz(frames: &[Vec<f32>], n_bars: usize, lo: f32, hi: f32) -> f32 {
        let row = &frames[frames.len() / 2];
        let (peak_bar, &peak) = row.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        let half = peak * 0.5;
        let cross = |inside: usize, outside: usize| -> f32 {
            let (fi, fo) = (bar_center_freq(inside, n_bars, lo, hi),
                            bar_center_freq(outside, n_bars, lo, hi));
            let (vi, vo) = (row[inside], row[outside]);
            if (vi - vo).abs() < 1e-9 { return fo; }
            fi + (fo - fi) * (vi - half) / (vi - vo)
        };
        let mut l = peak_bar;
        while l > 0 && row[l] > half { l -= 1; }
        let mut r = peak_bar;
        while r + 1 < n_bars && row[r] > half { r += 1; }
        let lo_hz = if l < peak_bar { cross(l + 1, l) } else { bar_center_freq(l, n_bars, lo, hi) };
        let hi_hz = if r >= 1 { cross(r - 1, r) } else { bar_center_freq(r, n_bars, lo, hi) };
        hi_hz - lo_hz
    }

    // ── The question this module has to answer ──────────────────────────────

    /// Does it beat the FFT?
    ///
    /// Two tones 1.5 Hz apart at 60 Hz. The FFT path resolves sigma ~3.5 Hz
    /// flat, so it renders them as one blob at any setting — its window is fixed
    /// near 171 ms and no bar mapping recovers what the window threw away.
    /// Constant-Q at 60 Hz wants sigma 0.42 Hz, which needs a window the FFT
    /// never uses. This is the case the transform exists for.
    #[test]
    fn resolves_bass_detail_the_fft_cannot() {
        let (lo, hi) = (40.0, 90.0);
        let n_bars = 120;
        let grid = grid_q(n_bars, lo, hi);
        // The wavelet runs ~3 s here, so the signal has to be longer than that
        // or truncation widens the very peak being measured.
        let sig = tone(&[60.0, 61.5], 6.0, SR_B);

        let c = cfg(1.0, 5.0, 5, 0.5);
        let q = effective_q_at(60.0, grid, &c);
        let sigma = 60.0 / q;
        let fft_sigma = fft_sigma_hz(SR, FFT_N);
        println!(
            "60 Hz — superlet sigma {sigma:.2} Hz (Q {q:.0}, window {:.2} s) vs FFT sigma {fft_sigma:.2} Hz",
            window_seconds_at(60.0, grid, &c),
        );
        assert!(sigma < fft_sigma / 3.0, "superlet sigma {sigma} not much finer than FFT {fft_sigma}");

        // And it must actually show two peaks, not merely claim the resolution.
        let fr = analyze(&sig, SR_B, n_bars, lo, hi, hop_for_fps(SR_B, 10.0), &c);
        let a = mag_at(&fr, 60.0, n_bars, lo, hi);
        let b = mag_at(&fr, 61.5, n_bars, lo, hi);
        let mid = mag_at(&fr, 60.75, n_bars, lo, hi);
        let dip = mid / a.min(b).max(1e-9);
        println!("  1.5 Hz split at 60 Hz — dip ratio {dip:.3}");
        assert!(dip < 0.75, "60/61.5 Hz not separated: dip {dip}");
    }

    /// The other half of the claim: above the crossover the FFT already
    /// out-resolves the display, so the win has to be time. The superlet window
    /// must come out well under the FFT's fixed 171 ms up there.
    #[test]
    fn beats_the_fft_on_time_in_the_treble() {
        let n_bars = 1024;
        let grid = grid_q(n_bars, 20.0, 24_000.0);
        let fft_window = FFT_N as f32 / SR as f32;
        for &f in &[4_000.0f32, 10_000.0, 16_000.0] {
            let w = window_seconds_at(f, grid, &AsltPreset::Standard.config());
            println!("{f:.0} Hz — superlet window {:.1} ms vs FFT {:.1} ms",
                     w * 1000.0, fft_window * 1000.0);
            assert!(w < fft_window * 0.5, "{f} Hz window {w} not much shorter than {fft_window}");
        }
    }

    /// Where the crossover sits.
    #[test]
    fn crossover_is_where_the_maths_says() {
        let grid = grid_q(1024, 20.0, 24_000.0);
        let x = fft_crossover_hz(SR, FFT_N, grid);
        println!("grid Q {grid:.0}, FFT sigma {:.2} Hz, crossover {x:.0} Hz",
                 fft_sigma_hz(SR, FFT_N));
        assert!((400.0..700.0).contains(&x), "crossover {x} Hz is not where it should be");
    }

    // ── Frequency-domain route ──────────────────────────────────────────────

    /// The whole point: the fast route must give the same picture as the slow
    /// one. Anything else is losing pixels to buy speed.
    ///
    /// Checked against a broadband signal — tones, a sweep and noise — so
    /// nothing hides in a spectrum that happens to be empty where the two
    /// disagree.
    #[test]
    fn fft_route_matches_direct() {
        let sr = SR_A;
        let n = (2.0 * sr as f32) as usize;
        let tau = std::f32::consts::TAU;
        let mut seed = 0x2545F491_4F6CDD1Du64;
        let sig: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let noise = ((seed >> 40) as f32 / 8_388_608.0) - 1.0;
                0.4 * (tau * 110.0 * t).sin()
                    + 0.3 * (tau * (300.0 + 200.0 * t) * t).sin()
                    + 0.2 * (tau * 2_500.0 * t).sin()
                    + 0.1 * noise
            })
            .collect();

        let hop = hop_for_fps(sr, 60.0);
        let frames = frame_count(sig.len(), hop);
        let cfg = cfg(1.0, 1.5, 4, 0.5);
        let (lo, hi) = (40.0, 6_000.0);
        let n_bars = 96;
        let grid = grid_q(n_bars, lo, hi);

        // Forced, not gated on `fft_is_cheaper`: correctness must not depend on a
        // performance heuristic, or retuning the threshold silently stops
        // testing the thing it was meant to protect.
        let mut compared = 0usize;
        let mut worst = 0.0f32;
        for bar in 0..n_bars {
            let f = bar_center_freq(bar, n_bars, lo, hi);
            let sl = Superlet::new(f, effective_q_at(f, grid, &cfg), &cfg, sr as f32);
            for w in &sl.wavelets {
                let Some(fast) = w.magnitudes_via_fft(&sig, hop, frames) else { continue };
                compared += 1;
                let slow: Vec<f32> = (0..frames)
                    .map(|fi| w.response(&sig, (fi * hop) as isize))
                    .collect();
                let peak = slow.iter().cloned().fold(0.0f32, f32::max).max(1e-12);
                for (a, b) in fast.iter().zip(slow.iter()) {
                    worst = worst.max((a - b).abs() / peak);
                }
            }
        }
        println!("{compared} wavelets compared; worst relative error {worst:.2e}");
        assert!(compared >= 20, "only {compared} wavelets tested — proves little");
        assert!(worst < 2e-3, "FFT route drifts from direct by {worst}");
    }

    /// The threshold must actually route the expensive bars through the
    /// transform at settings someone would really use.
    #[test]
    fn threshold_routes_the_expensive_bars() {
        let hop = hop_for_fps(SR, 180.0);
        let grid = grid_q(1024, 20.0, 24_000.0);
        let c = AsltPreset::Extreme.config();
        let mut routed = 0;
        for bar in 0..1024 {
            let f = bar_center_freq(bar, 1024, 20.0, 24_000.0);
            let sl = Superlet::new(f, effective_q_at(f, grid, &c), &c, SR as f32);
            if sl.prefers_fft(hop) { routed += 1; }
        }
        println!("Extreme @ 48 kHz/180 fps: {routed} of 1024 bars take the FFT route");
        assert!(routed > 100, "only {routed} bars routed — threshold is too strict");
        assert!(routed < 900, "{routed} bars routed — threshold is too loose");
    }

    /// Edge frames must stay on the direct path, renormalisation and all — a
    /// convolution cannot express it and would reintroduce the 6 dB edge dip.
    #[test]
    fn fft_route_keeps_edge_behaviour() {
        let sr = SR_A;
        let f = 80.0f32;
        let sig = tone(&[f], 2.0, sr);
        let hop = hop_for_fps(sr, 60.0);
        let frames = frame_count(sig.len(), hop);
        let cfg = cfg(1.0, 1.5, 4, 0.5);
        let sl = Superlet::new(f, 60.0, &cfg, sr as f32);
        assert!(sl.prefers_fft(hop), "test needs a bar that takes the FFT route");

        let fast = sl.responses(&sig, hop, frames);
        let slow: Vec<f32> = (0..frames)
            .map(|fi| sl.response(&sig, (fi * hop) as isize))
            .collect();
        // First and last frames are pure edge cases.
        for i in [0usize, 1, frames - 2, frames - 1] {
            assert!((fast[i] - slow[i]).abs() < 1e-4,
                    "edge frame {i}: fast {} vs direct {}", fast[i], slow[i]);
        }
        assert!(fast[0] > 0.7, "edge amplitude collapsed to {}", fast[0]);
    }

    /// The shared-signal route must give the same answer as the per-kernel one.
    ///
    /// This is the whole safety net for sharing the forward transform: the
    /// block decomposition changes, so the outputs land in different places
    /// within each block and the valid ranges differ. If any of that indexing
    /// is off the result is not a crash, it is a plausible-looking spectrum.
    #[test]
    fn shared_signal_route_matches_per_kernel() {
        let sr = SR_A;
        let sig = tone(&[60.0, 61.5, 200.0], 3.0, sr);
        let hop = hop_for_fps(sr, 60.0);
        let frames = frame_count(sig.len(), hop);

        // Several kernel lengths, so more than one block size is exercised and
        // the stride differs from `n - k + 1` by varying amounts.
        for &(f, c) in &[(60.0f32, 40.0f32), (60.0, 90.0), (120.0, 60.0), (200.0, 150.0)] {
            let w = Morlet::new(f, c, sr as f32);
            let k = w.re.len();
            let n = fft_block_len(k);
            if n <= k { continue; }
            let want = w.magnitudes_via_fft(&sig, hop, frames)
                .unwrap_or_else(|| panic!("f={f} c={c}: per-kernel route declined"));
            let blocks = SignalBlocks::build(&sig, n);
            let got = w.magnitudes_via_shared(&sig, &blocks, hop, frames)
                .unwrap_or_else(|| panic!("f={f} c={c}: shared route declined"));

            let peak = want.iter().cloned().fold(0.0f32, f32::max).max(1e-12);
            let worst = got.iter().zip(&want)
                .map(|(a, b)| (a - b).abs() / peak)
                .fold(0.0f32, f32::max);
            println!("f={f} c={c} k={k} n={n} stride {} vs {}: worst {worst:.2e}",
                     blocks.stride, n - k + 1);
            assert!(worst < 2e-5, "f={f} c={c}: routes differ by {worst:.3e}");
        }
    }

    /// A stride longer than the kernel allows must be refused, not silently
    /// leave gaps between the valid regions of consecutive blocks.
    #[test]
    fn a_stride_too_long_for_the_kernel_is_refused() {
        let sr = SR_A;
        let sig = tone(&[100.0], 2.0, sr);
        let hop = hop_for_fps(sr, 60.0);
        let frames = frame_count(sig.len(), hop);
        let w = Morlet::new(100.0, 60.0, sr as f32);
        let n = fft_block_len(w.re.len());
        // Hand it blocks whose stride is the full transform length: with any
        // kernel longer than one tap that cannot cover every output.
        let bad = SignalBlocks { n, stride: n, data: SignalBlocks::build(&sig, n).data };
        assert!(w.magnitudes_via_shared(&sig, &bad, hop, frames).is_none());
    }

    /// `analyze` must give the same spectrum whether or not a GPU took part.
    ///
    /// The per-pipeline test compares one block size in isolation. This one
    /// covers the wiring around it: which bars are routed where, the chunking,
    /// the edge frames the GPU hands back for the CPU to fill, and the
    /// recombination of members that went down different paths. A mistake in
    /// any of those yields a spectrum, just not the right one.
    /// The device boundary the GPU/CPU comparison has always used: low enough
    /// that the bass bars really do take the device branch on a signal this
    /// short, where the shipped threshold is tuned for real tracks.
    ///
    /// Injected rather than stored in a global, so a test running in parallel
    /// cannot observe or move it.
    fn eligible_gpu(n: usize) -> bool {
        n >= 1 << 13
    }
    /// Route nothing to the device.
    fn never_gpu(_n: usize) -> bool {
        false
    }

    /// Run an analysis with the route stated outright.
    #[allow(clippy::too_many_arguments)]
    fn analyze_via(
        route: RouteDecision,
        sig: &[f32],
        sr: u32,
        bars: usize,
        lo: f32,
        hi: f32,
        hop: usize,
        cfg: &AsltConfig,
    ) -> Vec<Vec<f32>> {
        analyze_routed(sig, sr, bars, lo, hi, hop, cfg, &|| true, &|_| {}, route)
    }

    #[test]
    fn analyze_agrees_with_and_without_the_gpu() {
        if super::super::gpu::GpuFft::shared().is_none() {
            println!("no GPU adapter — skipping");
            return;
        }
        let sr = 16_000u32;
        let tau = std::f32::consts::TAU;
        let n = (4.0 * sr as f32) as usize;
        let sig: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.5 * (tau * 45.0 * t).sin()
                    + 0.3 * (tau * 180.0 * t).sin()
                    + 0.2 * (tau * 900.0 * t).sin()
            })
            .collect();
        let hop = hop_for_fps(sr, 60.0);
        let (lo, hi) = (20.0f32, 8_000.0);
        let bars = 128;
        let cfg = cfg(1.0, 1.2, 4, 0.5);

        // Stated per call rather than by moving a shared threshold: the old
        // version wrote to a process-wide atomic that another test's analysis
        // could read mid-run.
        let with = analyze_via(eligible_gpu, &sig, sr, bars, lo, hi, hop, &cfg);
        let without = analyze_via(never_gpu, &sig, sr, bars, lo, hi, hop, &cfg);

        assert_eq!(with.len(), without.len(), "frame counts differ");
        assert!(!with.is_empty());
        let peak = without.iter().flatten().cloned().fold(0.0f32, f32::max).max(1e-12);
        let worst = with.iter().flatten().zip(without.iter().flatten())
            .map(|(a, b)| (a - b).abs() / peak)
            .fold(0.0f32, f32::max);
        println!("analyze with/without GPU: worst relative difference {worst:.2e}");
        assert!(worst < 1e-3, "GPU and CPU spectra differ by {worst:.3e}");
    }

    /// The whole safety case for adding a second bar scale: with `Log` selected
    /// the transform must produce exactly what it produced before the scale
    /// existed — not close, exactly. Anything else means every cached analysis
    /// and every measurement in this repository was taken against different
    /// maths.
    #[test]
    fn the_log_scale_changes_nothing_at_all() {
        let sr = SR_A;
        let sig = tone(&[60.0, 61.5, 120.0], 4.0, sr);
        let hop = hop_for_fps(sr, 60.0);
        let (lo, hi) = (50.0, 200.0);
        let n_bars = 24;
        let cfg = AsltPreset::Fast.config();
        assert_eq!(cfg.scale, super::super::freq_scale::FreqScale::Log,
                   "Log has to stay the default, or existing caches silently change meaning");

        // The pre-scale code path, reconstructed: one scalar grid_q for every
        // bar, and `bar_center_freq` for the centres.
        let grid = grid_q(n_bars, lo, hi);
        let want: Vec<(f32, f32)> = (0..n_bars)
            .map(|b| {
                let f = bar_center_freq(b, n_bars, lo, hi);
                (f, effective_q_at(f, grid, &cfg))
            })
            .collect();
        let got: Vec<(f32, f32)> = (0..n_bars)
            .map(|b| {
                let f = cfg.scale.bar_center(b, n_bars, lo, hi);
                (f, effective_q_at(f, cfg.scale.grid_q_at(b, n_bars, lo, hi), &cfg))
            })
            .collect();
        assert_eq!(want, got, "the Log scale must be the old code, bit for bit");

        // And end to end, since the plan is built from those two numbers.
        //
        // Both calls are pinned to the same route. Exact equality is only
        // meaningful within one route -- GPU and CPU agree to a tolerance, not
        // bit for bit -- and this used to depend on a global threshold another
        // test could move between these two lines, which made it fail
        // intermittently under a parallel suite and pass in isolation.
        let a = analyze_via(never_gpu, &sig, sr, n_bars, lo, hi, hop, &cfg);
        let b = analyze_via(never_gpu, &sig, sr, n_bars, lo, hi, hop, &cfg);
        assert_eq!(a, b, "analysis must be deterministic before anything else is claimed");
    }

    /// ERB has to actually work, not merely compile: a tone must land where it
    /// belongs, and the bass must cost less than it does on the log grid.
    #[test]
    fn the_erb_scale_resolves_and_costs_less_in_the_bass() {
        use super::super::freq_scale::FreqScale;
        let sr = SR_A;
        let (lo, hi) = (20.0, 20_000.0);
        let n_bars = 128;
        let hop = hop_for_fps(sr, 60.0);
        let mut cfg = AsltPreset::Fast.config();
        cfg.scale = FreqScale::Erb;

        // A tone lands on the bar nearest its frequency, same as on log.
        let f = FreqScale::Erb.bar_center(n_bars / 2, n_bars, lo, hi);
        let fr = analyze(&tone(&[f], 2.0, sr), sr, n_bars, lo, hi, hop, &cfg);
        assert!(!fr.is_empty());
        let mid = fr.len() / 2;
        let peak = (0..n_bars)
            .max_by(|&a, &b| fr[mid][a].partial_cmp(&fr[mid][b]).unwrap())
            .unwrap();
        assert!(
            (peak as isize - (n_bars / 2) as isize).abs() <= 1,
            "tone at {f:.1} Hz peaked at bar {peak}, expected {}", n_bars / 2,
        );

        // The bass is cheaper, which is the reason to want this at all.
        // The bottom eighth of the display, which is where log spacing spends
        // bars on resolution the window cap cannot deliver.
        let taps = |c: &AsltConfig| -> f64 {
            bar_taps_per_frame(sr, n_bars, lo, hi, c).iter().take(n_bars / 8).sum()
        };
        let mut log_cfg = cfg.clone();
        log_cfg.scale = FreqScale::Log;
        let (erb_cost, log_cost) = (taps(&cfg), taps(&log_cfg));
        assert!(
            erb_cost < log_cost * 0.75,
            "ERB bass should cost clearly less: {erb_cost:.0} vs {log_cost:.0} taps/frame",
        );
    }

    /// The routes must also agree end-to-end, not just per superlet.
    #[test]
    fn analyze_is_route_independent() {
        let sr = SR_A;
        let sig = tone(&[60.0, 61.5, 120.0], 4.0, sr);
        let hop = hop_for_fps(sr, 60.0);
        let (lo, hi) = (50.0, 200.0);
        let n_bars = 16;

        // Few bars keeps the direct comparison affordable, but few bars also
        // means a low grid Q and therefore short kernels — so sharpness is
        // pushed past the grid to get kernels long enough to route.
        let c = cfg(8.0, 5.0, 4, 0.5);
        {
            let grid = grid_q(n_bars, lo, hi);
            let f = bar_center_freq(0, n_bars, lo, hi);
            let sl = Superlet::new(f, effective_q_at(f, grid, &c), &c, sr as f32);
            assert!(sl.prefers_fft(hop), "test would not exercise the FFT route");
        }
        let fast = analyze(&sig, sr, n_bars, lo, hi, hop, &c);
        let slow = analyze_direct_only(&sig, sr, n_bars, lo, hi, hop, &c);
        let peak = slow.iter().flatten().cloned().fold(0.0f32, f32::max).max(1e-12);
        let worst = fast.iter().flatten().zip(slow.iter().flatten())
            .map(|(a, b)| (a - b).abs() / peak)
            .fold(0.0f32, f32::max);
        println!("end-to-end worst relative error {worst:.2e}");
        assert!(worst < 2e-3, "analyze() differs by route: {worst}");
    }

    /// `analyze` with every superlet forced down the direct path, for comparison.
    #[allow(clippy::too_many_arguments)]
    fn analyze_direct_only(
        signal: &[f32], sample_rate: u32, n_bars: usize,
        min_freq: f32, max_freq: f32, hop: usize, cfg: &AsltConfig,
    ) -> Vec<Vec<f32>> {
        let frames = frame_count(signal.len(), hop);
        let grid = grid_q(n_bars, min_freq, max_freq);
        let nyquist = sample_rate as f32 * 0.5;
        let mut out = vec![vec![0.0f32; n_bars]; frames];
        for bar in 0..n_bars {
            let f = bar_center_freq(bar, n_bars, min_freq, max_freq);
            if f >= nyquist { continue; }
            let sl = Superlet::new(f, effective_q_at(f, grid, cfg), cfg, sample_rate as f32);
            for (fi, row) in out.iter_mut().enumerate() {
                row[bar] = sl.response(signal, (fi * hop) as isize);
            }
        }
        out
    }

    // ── Parameterisation ────────────────────────────────────────────────────

    /// The efficiency claim that motivated the rewrite: the paper's layout keeps
    /// 0.577 of its longest wavelet's Q, a narrowed spread keeps far more, for
    /// identical time cost.
    #[test]
    fn narrow_spread_keeps_more_resolution_per_cycle() {
        let paper = q_per_cycle(&cfg(1.0, 1.0, 8, 0.0));
        let narrow = q_per_cycle(&cfg(1.0, 1.0, 8, 0.5));
        let single = q_per_cycle(&cfg(1.0, 1.0, 1, 0.0));
        println!("q_per_cycle — paper {paper:.3}, spread 0.5 {narrow:.3}, single {single:.3}");
        assert!((paper - 0.577).abs() < 0.06, "paper layout should sit near 0.577, got {paper}");
        assert!(narrow > paper * 1.25, "narrowing should buy over 25 percent: {paper} to {narrow}");
        assert!((single - 1.0).abs() < 1e-6);
    }

    /// `full_detail_above` restates `max_window_s` as the thing it decides, so
    /// it has to agree with what the transform actually does — and with the
    /// claims the preset list makes to the user.
    #[test]
    fn full_detail_frequency_matches_the_transform() {
        let grid = grid_q(1024, 20.0, 24_000.0);

        for p in [AsltPreset::Fast, AsltPreset::Standard, AsltPreset::High,
                  AsltPreset::Ultra, AsltPreset::Extreme] {
            let c = p.config();
            let f = full_detail_above(grid, &c);
            // At the quoted frequency the cap must be just about to bind: full
            // grid Q at and above it, less below.
            assert!(
                (effective_q_at(f * 1.02, grid, &c) - grid * c.q_ratio).abs() < grid * 0.02,
                "{}: {f:.0} Hz should already be grid-exact", p.label(),
            );
            assert!(
                effective_q_at(f * 0.7, grid, &c) < grid * c.q_ratio * 0.95,
                "{}: below {f:.0} Hz the window cap should bind", p.label(),
            );
            // And the inverse must land back on the same window.
            let mut c2 = c.clone();
            c2.max_window_s = window_for_full_detail_above(f, grid, &c);
            assert!(
                (c2.max_window_s - c.max_window_s).abs() < c.max_window_s * 0.01,
                "{}: round trip gave {} s, not {} s",
                p.label(), c2.max_window_s, c.max_window_s,
            );
        }

        // The preset list tells the user High "clearly beats the FFT below
        // 200 Hz" and Extreme is "grid-exact to ~35 Hz". Those are the numbers
        // someone picks a preset by, so they should not drift from the maths.
        let high = full_detail_above(grid, &AsltPreset::High.config());
        assert!((150.0..260.0).contains(&high), "High resolves fully above {high:.0} Hz");
        let extreme = full_detail_above(grid, &AsltPreset::Extreme.config());
        assert!((30.0..55.0).contains(&extreme), "Extreme resolves fully above {extreme:.0} Hz");

        // Longer window must always reach lower — the whole point of the knob.
        let mut a = AsltPreset::High.config();
        let f_short = full_detail_above(grid, &a);
        a.max_window_s *= 2.0;
        assert!(full_detail_above(grid, &a) < f_short * 0.6);
    }

    /// The window cap must limit the bass and stop mattering in the treble.
    #[test]
    fn window_cap_binds_low_and_releases_high() {
        let grid = grid_q(1024, 20.0, 24_000.0);
        let c = cfg(1.0, 1.0, 5, 0.5);
        assert!(effective_q_at(20.0, grid, &c) < grid * 0.5);
        assert!(window_seconds_at(20.0, grid, &c) <= 1.01);
        assert!((effective_q_at(8_000.0, grid, &c) - grid).abs() < 1.0);
        assert!(window_seconds_at(8_000.0, grid, &c) < 0.05);
    }

    /// Presets must be monotone in cost and never exceed their own budget.
    #[test]
    fn ladder_is_monotone_and_within_budget() {
        let grid = grid_q(1024, 20.0, 24_000.0);
        let mut prev = 0.0f64;
        for p in [
            AsltPreset::Fast, AsltPreset::Standard, AsltPreset::High,
            AsltPreset::Ultra, AsltPreset::Extreme,
        ] {
            let c = p.config();
            let taps: f64 = bar_taps_per_frame(SR, 1024, 20.0, 24_000.0, &c).iter().sum();
            let w20 = window_seconds_at(20.0, grid, &c);
            println!("{:<9} bass window {:.2} s, {:.3e} taps/frame", p.label(), w20, taps);
            assert!(w20 <= c.max_window_s * 1.02, "{} exceeded its own budget", p.label());
            assert!(taps > prev, "{} not costlier than the rung below", p.label());
            prev = taps;
        }
    }

    /// Grid-exact must mean grid-exact: measured peak width has to match what
    /// `effective_q_at` promises, or the UI is quoting fiction.
    #[test]
    fn measured_width_matches_the_promise() {
        let (lo, hi) = (700.0, 1400.0);
        let n_bars = 128;
        let grid = grid_q(n_bars, lo, hi);
        let c = cfg(1.0, 2.0, 5, 0.5);
        let f = nearest_bar_freq(1000.0, n_bars, lo, hi);
        // Signal comfortably longer than the ~0.3 s window, so the measurement
        // is of the wavelet and not of the signal running out.
        let fr = analyze(&tone(&[f], 2.0, SR_A), SR_A, n_bars, lo, hi, hop_for_fps(SR_A, 60.0), &c);
        let measured = half_max_width_hz(&fr, n_bars, lo, hi);
        let sigma = f / effective_q_at(f, grid, &c);
        let predicted = 2.0 * (2.0f32 * 2.0f32.ln()).sqrt() * sigma;
        let ratio = measured / predicted;
        println!("{f:.0} Hz — measured FWHM {measured:.2} Hz, predicted {predicted:.2} Hz");
        assert!((0.6..1.6).contains(&ratio), "measured {measured} vs predicted {predicted}");
    }

    // ── Invariants carried over from the first cut ──────────────────────────

    #[test]
    fn unit_sine_reads_unity_at_any_setting() {
        let (lo, hi) = (200.0, 5000.0);
        let f = nearest_bar_freq(1000.0, 128, lo, hi);
        for c in [cfg(0.5, 0.3, 3, 0.5), cfg(1.0, 1.0, 5, 0.5), cfg(2.0, 3.0, 7, 0.6)] {
            let fr = analyze(&tone(&[f], 1.0, SR_A), SR_A, 128, lo, hi, hop_for_fps(SR_A, 60.0), &c);
            let m = mag_at(&fr, f, 128, lo, hi);
            assert!((m - 1.0).abs() < 0.06, "unit sine read {m} for {c:?}");
        }
    }

    #[test]
    fn magnitude_scales_with_amplitude() {
        let (lo, hi) = (200.0, 5000.0);
        let c = cfg(1.0, 1.0, 4, 0.5);
        let hop = hop_for_fps(SR_A, 60.0);
        let loud = tone(&[1000.0], 0.5, SR);
        let quiet: Vec<f32> = loud.iter().map(|s| s * 0.5).collect();
        let a = mag_at(&analyze(&loud, SR_A, 128, lo, hi, hop, &c), 1000.0, 128, lo, hi);
        let b = mag_at(&analyze(&quiet, SR_A, 128, lo, hi, hop, &c), 1000.0, 128, lo, hi);
        assert!((a / b - 2.0).abs() < 0.1, "expected 2:1, got {a}:{b}");
    }

    #[test]
    fn rejects_off_frequency_energy() {
        let (lo, hi) = (200.0, 5000.0);
        let c = cfg(1.0, 1.0, 5, 0.5);
        let f = nearest_bar_freq(1000.0, 512, lo, hi);
        let fr = analyze(&tone(&[f], 0.6, SR_A), SR_A, 512, lo, hi, hop_for_fps(SR_A, 60.0), &c);
        let on = mag_at(&fr, f, 512, lo, hi);
        let off = mag_at(&fr, 2000.0, 512, lo, hi);
        assert!(on > off * 100.0, "selectivity too weak: on={on} off={off}");
    }

    /// Truncating at the edge halves the effective window, so some loss is
    /// expected; what must not happen is the 6 dB cliff zero-padding gives or
    /// the near-total cancellation mirroring gives.
    #[test]
    fn edge_frames_keep_their_amplitude() {
        let (lo, hi) = (200.0, 5000.0);
        let c = cfg(1.0, 1.0, 4, 0.5);
        let f = nearest_bar_freq(1000.0, 128, lo, hi);
        let fr = analyze(&tone(&[f], 0.6, SR_A), SR_A, 128, lo, hi, hop_for_fps(SR_A, 60.0), &c);
        let bar = (0..128)
            .min_by(|&a, &b| {
                let da = (bar_center_freq(a, 128, lo, hi) - f).abs();
                let db = (bar_center_freq(b, 128, lo, hi) - f).abs();
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        assert!(fr[0][bar] > 0.7, "first frame dipped to {}", fr[0][bar]);
    }

    #[test]
    fn silence_stays_finite() {
        let c = AsltPreset::Extreme.config();
        let fr = analyze(&vec![0.0f32; SR_A as usize / 2], SR_A, 128, 20.0, 20_000.0,
                         hop_for_fps(SR_A, 60.0), &c);
        for &v in fr.iter().flatten() {
            assert!(v.is_finite() && v < 1e-3, "silence produced {v}");
        }
    }

    #[test]
    fn above_nyquist_reads_zero() {
        let c = AsltPreset::Fast.config();
        let n_bars = 256;
        let fr = analyze(&tone(&[1000.0], 0.4, SR), SR_A, n_bars, 20.0, 40_000.0,
                         hop_for_fps(SR_A, 60.0), &c);
        let mid = &fr[fr.len() / 2];
        for (bar, &v) in mid.iter().enumerate() {
            if bar_center_freq(bar, n_bars, 20.0, 40_000.0) >= SR_A as f32 * 0.5 {
                assert_eq!(v, 0.0);
            }
        }
    }

    #[test]
    fn abort_stops_work_rather_than_discarding_it() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let c = cfg(1.0, 1.0, 5, 0.5);
        let n_bars = 256;
        let started = AtomicUsize::new(0);
        let out = analyze_with_progress(
            &tone(&[1000.0], 1.0, SR), SR_A, n_bars, 200.0, 5000.0,
            hop_for_fps(SR_A, 60.0), &c,
            &|| started.load(Ordering::Relaxed) < 8,
            &|_| { started.fetch_add(1, Ordering::Relaxed); },
        );
        assert!(out.is_empty(), "cancelled run should yield nothing");
        let ran = started.load(Ordering::Relaxed);
        assert!(ran < n_bars / 2, "{ran} of {n_bars} bars ran despite cancelling");
    }

    #[test]
    fn cost_model_tracks_actual_work() {
        let cheap: f64 = bar_taps_per_frame(SR, 128, 20.0, 20_000.0, &AsltPreset::Fast.config())
            .iter().sum();
        let dear: f64 = bar_taps_per_frame(SR, 128, 20.0, 20_000.0, &AsltPreset::Extreme.config())
            .iter().sum();
        assert!(dear > cheap * 3.0, "Extreme should dwarf Fast: {cheap} vs {dear}");
    }

    /// Why Q must come from the grid alone, and never from a band's content.
    ///
    /// Changing one band's Q changes what that band reads on broadband material
    /// but not on a tone, because the wavelet is normalised so a tone reads 1.0
    /// whatever its bandwidth while noise reads proportional to √bandwidth. A
    /// band whose Q differs from its neighbours' therefore draws as a bright
    /// stripe, and no single per-band gain can correct it — the correction a
    /// tone needs is 0 dB and the correction noise needs is not.
    ///
    /// This is what killed the "follow vibrato" adaptation, which cut Q as far as
    /// 0.30 on real material and left +9.5 dB steps across the waterfall. Keep
    /// this test as the reason: anything that reintroduces per-band Q variation
    /// has to answer it first.
    #[test]
    fn q_change_level_law() {
        let sr = SR;
        let secs = 4.0f32;
        let n = (secs * sr as f32) as usize;
        let f = 1000.0f32;
        let tau = std::f32::consts::TAU;

        let pure = tone(&[f], secs, sr);
        let mut seed = 0xC0FFEEu32;
        let noise: Vec<f32> = (0..n).map(|_| {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        }).collect();
        let vib: Vec<f32> = { // vibrato'd tone: the case adaptation targets
            let mut p = 0.0f32;
            (0..n).map(|i| {
                let t = i as f32 / sr as f32;
                p += tau * f * (1.0 + 0.025 * (tau * 5.5 * t).sin()) / sr as f32;
                p.sin()
            }).collect()
        };

        let cfg = AsltPreset::Standard.config();
        let q_free = 144.0f32;
        println!("\n{:>7} {:>9} {:>9} {:>9}", "q/qfree", "tone dB", "noise dB", "vib dB");
        let level = |sig: &[f32], q: f32| {
            let sl = Superlet::new(f, q, &cfg, sr as f32);
            let s: f64 = (0..40).map(|i| sl.response(sig, (n / 2 + i * 64) as isize) as f64).sum();
            20.0 * (s / 40.0).max(1e-12).log10()
        };
        let (t0, n0, v0) = (level(&pure, q_free), level(&noise, q_free), level(&vib, q_free));
        for r in [1.0f32, 0.78, 0.5, 0.3] {
            let q = q_free * r;
            let (dt, dn, dv) =
                (level(&pure, q) - t0, level(&noise, q) - n0, level(&vib, q) - v0);
            println!("{r:>7.2} {dt:>+9.2} {dn:>+9.2} {dv:>+9.2}");
            // A tone is indifferent to Q; broadband content is not. Those two
            // facts together are what makes the step uncorrectable.
            assert!(dt.abs() < 0.1, "tone level moved with Q: {dt:+.2} dB at r={r}");
            if r < 0.9 {
                assert!(dn > 0.5, "noise level should rise as Q falls: {dn:+.2} dB at r={r}");
            }
        }
        println!("(0 dB = no step. Differing columns => no single gain fixes it.)");
    }

    /// Where does a transient actually land in each band, and how wide is it?
    ///
    /// The question behind the slanted-kick report: is a low band *late*, which
    /// a per-band time shift would fix, or merely *wide*, which it would not?
    ///
    /// `cargo test --release -- --ignored --nocapture transient_alignment`
    #[test]
    #[ignore = "measurement — run explicitly"]
    fn transient_alignment() {
        let sr = SR;
        let secs = 3.0f32;
        let n = (secs * sr as f32) as usize;
        let hit = (1.5 * sr as f32) as usize;

        // A click: flat across every band, so each band's response is exactly
        // that band's own impulse response, with nothing from the signal's own
        // spectrum confusing the timing.
        let mut sig = vec![0.0f32; n];
        sig[hit] = 1.0;

        let hop = hop_for_fps(sr, 180.0);
        let n_bars = 1024;
        let (lo, hi) = (20.0f32, 24_000.0);
        let grid = grid_q(n_bars, lo, hi);
        let cfg = AsltPreset::High.config();
        let frames = analyze(&sig, sr, n_bars, lo, hi, hop, &cfg);
        assert!(!frames.is_empty());

        let hit_ms = 1000.0 * hit as f64 / sr as f64;
        let frame_ms = 1000.0 * hop as f64 / sr as f64;
        println!("\nclick at {hit_ms:.0} ms, {frame_ms:.2} ms per frame, preset High");
        println!("{:>9} {:>10} {:>10} {:>10} {:>10}",
                 "freq", "peak ms", "onset ms", "width ms", "window ms");

        for &f in &[30.0f32, 60.0, 120.0, 250.0, 500.0, 1000.0, 4000.0, 12000.0] {
            // Nearest bar to this frequency.
            let bar = (0..n_bars)
                .min_by(|&a, &b| {
                    let da = (bar_center_freq(a, n_bars, lo, hi) - f).abs();
                    let db = (bar_center_freq(b, n_bars, lo, hi) - f).abs();
                    da.partial_cmp(&db).unwrap()
                }).unwrap();

            let col: Vec<f32> = frames.iter().map(|fr| fr[bar]).collect();
            let peak = col.iter().cloned().fold(0.0f32, f32::max);
            if peak <= 0.0 { continue; }
            let peak_i = col.iter().position(|&v| v == peak).unwrap();
            // Onset: first frame reaching half the peak — this is what the eye
            // reads as "the band lit up", and it is not the same as the peak.
            let onset_i = col.iter().position(|&v| v >= peak * 0.5).unwrap();
            let last_i = col.iter().rposition(|&v| v >= peak * 0.5).unwrap();

            println!("{:>9.0} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
                     bar_center_freq(bar, n_bars, lo, hi),
                     peak_i as f64 * frame_ms - hit_ms,
                     onset_i as f64 * frame_ms - hit_ms,
                     (last_i - onset_i) as f64 * frame_ms,
                     1000.0 * window_seconds_at(f, grid, &cfg) as f64);
        }
        println!("peak/onset are relative to the click: 0 = aligned, + = late.");
    }

    /// How much of the frequency-domain route is spent transforming the *signal*
    /// over and over?
    ///
    /// Overlap-save transforms the signal block by block for every kernel. The
    /// signal does not change between kernels, so every one of those after the
    /// first is repeated work — but only kernels that agree on both the block
    /// size *and* the block stride could share it, and the stride currently
    /// depends on the kernel length.
    ///
    /// `cargo test --release -- --ignored --nocapture signal_fft_redundancy`
    #[test]
    #[ignore = "measurement — run explicitly"]
    fn signal_fft_redundancy() {
        use std::collections::BTreeMap;
        let sr = SR;
        let secs = 300.0f32; // a five-minute track
        let n_sig = (secs * sr as f32) as usize;
        let hop = hop_for_fps(sr, 180.0);
        let n_bars = 1024;
        let (lo, hi) = (20.0f32, 24_000.0);
        let grid = grid_q(n_bars, lo, hi);

        println!("\n5-minute track at {sr} Hz, {n_bars} bars, 180 fps");
        println!("Transform counts, weighted by n·log₂n so sizes are comparable.");
        println!("{:<9} {:>9} {:>10} {:>10} {:>10} {:>10} {:>8}",
                 "preset", "fft wavs", "fwd now", "inv now", "fwd share", "inv share", "net");
        for preset in [AsltPreset::Standard, AsltPreset::High,
                       AsltPreset::Ultra, AsltPreset::Extreme] {
            let cfg = preset.config();
            // Cost of one transform of length n, in n·log2(n) units, so block
            // sizes can be added together meaningfully.
            let unit = |n: usize| (n as f64) * (n as f64).log2();

            let (mut fwd_now, mut inv_now, mut inv_share) = (0.0f64, 0.0f64, 0.0f64);
            let mut wavelets = 0u64;
            // One shared forward pass per distinct transform size. A stride of
            // n/2 is legal for every kernel of that size, since n >= 2k always.
            let mut per_size: BTreeMap<usize, f64> = BTreeMap::new();

            for bar in 0..n_bars {
                let f = bar_center_freq(bar, n_bars, lo, hi);
                if f >= sr as f32 * 0.5 { continue; }
                let q = effective_q_at(f, grid, &cfg);
                let sl = Superlet::new(f, q, &cfg, sr as f32);
                if !sl.prefers_fft(hop) { continue; }
                for w in &sl.wavelets {
                    let k = w.re.len();
                    let n = fft_block_len(k);
                    if n <= k { continue; }
                    wavelets += 1;
                    let blocks_now = n_sig.div_ceil(n - k + 1) as f64;
                    // A fixed n/2 stride is never longer than the current one,
                    // so sharing the forward pass costs *more* inverse passes.
                    let blocks_share = n_sig.div_ceil(n / 2) as f64;
                    fwd_now += blocks_now * unit(n);
                    inv_now += blocks_now * unit(n);
                    inv_share += blocks_share * unit(n);
                    per_size.entry(n).or_insert_with(|| n_sig.div_ceil(n / 2) as f64 * unit(n));
                }
            }
            let fwd_share: f64 = per_size.values().sum();
            let total_now = fwd_now + inv_now;
            let total_share = fwd_share + inv_share;
            let g = |v: f64| format!("{:.1}G", v / 1e9);
            println!("{:<9} {:>9} {:>10} {:>10} {:>10} {:>10} {:>8}",
                     preset.label(), wavelets, g(fwd_now), g(inv_now),
                     g(fwd_share), g(inv_share),
                     format!("{:.2}x", total_now / total_share.max(1.0)));
        }
        println!("Sharing removes nearly all forward work but adds inverse work,");
        println!("because one stride for every kernel of a size is shorter than each");
        println!("kernel's own. \"net\" is the honest end-to-end ratio.");
    }

    /// Benchmark. `cargo test --release -- --ignored --nocapture aslt_bench`
    #[test]
    #[ignore = "benchmark — run explicitly"]
    fn aslt_bench() {
        use std::time::Instant;
        // Long enough that the lowest bars' kernels are short *relative to the
        // signal*, as they are on a real track. At 5 s the bottom kernels are
        // nearly the whole signal, the frequency-domain route bails out, and the
        // benchmark measures a case that never happens in use.
        let secs = 20.0f32;
        let sig = tone(&[220.0, 1760.0], secs, SR);
        let hop = hop_for_fps(SR, 180.0);
        let grid = grid_q(1024, 20.0, 24_000.0);
        println!("\n{secs}s @ {SR} Hz, 1024 bars, 180 fps (hop {hop}), grid Q {grid:.0}");
        println!("{:<10} {:>8} {:>11} {:>9} {:>12}", "preset", "secs", "bass window", "Q@60Hz", "per 5min");
        for p in [
            AsltPreset::Fast, AsltPreset::Standard, AsltPreset::High,
            AsltPreset::Ultra, AsltPreset::Extreme,
        ] {
            let c = p.config();
            let t0 = Instant::now();
            let fr = analyze(&sig, SR, 1024, 20.0, 24_000.0, hop, &c);
            let el = t0.elapsed().as_secs_f64();
            assert!(!fr.is_empty());
            println!(
                "{:<10} {:>8.2} {:>10.2}s {:>9.0} {:>10.1} min",
                p.label(), el,
                window_seconds_at(60.0, grid, &c),
                effective_q_at(60.0, grid, &c),
                el * (300.0 / secs as f64) / 60.0,
            );
        }
    }
}


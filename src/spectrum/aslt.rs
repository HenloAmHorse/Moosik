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

/// The complex value every route produces before anything takes its magnitude.
///
/// Named because it is now a boundary rather than a local: a stereo analysis
/// combines the two channels *here*, before the magnitude, the log floor and
/// the geometric mean — none of which it could be done after.
pub type C32 = rustfft::num_complex::Complex<f32>;

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

/// Magnitudes of a complex column, in place of a second convolution.
fn norms(c: Vec<C32>) -> Vec<f32> {
    c.into_iter().map(|c| (c.re * c.re + c.im * c.im).sqrt()).collect()
}

/// `|(l + r) / 2|`, per frame — [`mix_complex`] and [`norms`] in one pass.
///
/// The same two expressions, applied to the same values in the same order. What
/// goes is the `Vec<C32>` between them, which on a joint run was one allocation
/// and one full pass per member per bar and was never read by anything else.
fn mix_norms(l: &[C32], r: &[C32]) -> Vec<f32> {
    l.iter()
        .zip(r)
        .map(|(a, b)| {
            let re = 0.5 * (a.re + b.re);
            let im = 0.5 * (a.im + b.im);
            (re * re + im * im).sqrt()
        })
        .collect()
}

/// `(l + r) / 2`, per frame — the mix, taken where it is still the mix.
///
/// The whole stereo design turns on this being applied to complex responses.
/// After the magnitude it would be the average of two magnitudes, which is a
/// different and wrong quantity: two channels in opposite polarity are both
/// loud and sum to silence, and only the complex form knows that.
fn mix_complex(l: &[C32], r: &[C32]) -> Vec<C32> {
    l.iter()
        .zip(r)
        .map(|(a, b)| C32 { re: 0.5 * (a.re + b.re), im: 0.5 * (a.im + b.im) })
        .collect()
}

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
        let (re, im, div) = self.response_parts(signal, centre);
        if div <= 0.0 { return 0.0; }
        (2.0 * (re * re + im * im).sqrt() / div) as f32
    }

    /// The same response, kept complex.
    ///
    /// This is the boundary a stereo analysis combines at. `M_j = (L_j + R_j)/2`
    /// is only the mix at this point: after the magnitude below, the average of
    /// two magnitudes has already discarded the phase that makes two channels
    /// cancel, and no later stage can put it back.
    ///
    /// The edge divisor is a property of the wavelet and the frame position, not
    /// of the signal, so it is identical for both channels — which is why
    /// averaging before it and averaging after it give the same answer, and why
    /// the combination is well defined at the signal edges too.
    fn response_c(&self, signal: &[f32], centre: isize) -> C32 {
        let (re, im, div) = self.response_parts(signal, centre);
        if div <= 0.0 { return C32 { re: 0.0, im: 0.0 }; }
        C32 { re: (2.0 * re / div) as f32, im: (2.0 * im / div) as f32 }
    }

    /// Raw f64 accumulators and the divisor the edges renormalise by.
    ///
    /// Returned unscaled and undivided so that [`Morlet::response`] can apply
    /// exactly the expression it always did — `2·|acc|/div` — and stay
    /// bit-identical to the magnitude this transform has produced since 1.4.
    /// The complex reader scales each component instead. One loop, two readers,
    /// and no second implementation of the convolution to drift.
    ///
    /// `div` is 1.0 in the interior, the surviving envelope sum at the edges,
    /// and 0.0 when the window misses the signal entirely.
    fn response_parts(&self, signal: &[f32], centre: isize) -> (f64, f64, f64) {
        let n = signal.len() as isize;
        let base = centre - self.half as isize;
        let end = base + self.re.len() as isize;
        let lo = base.max(0);
        let hi = end.min(n);
        if hi <= lo { return (0.0, 0.0, 0.0); }

        let mut acc_re = 0.0f64;
        let mut acc_im = 0.0f64;

        if base >= 0 && end <= n {
            // Interior: the envelope already sums to 1, so skip that accumulator.
            //
            // The window is taken once as a slice rather than indexed per tap.
            // The taps are visited in the same order and folded by the same two
            // expressions, so both f64 accumulators are bit-identical; what goes
            // is a bounds check on every one of the hundreds of thousands of
            // taps a bass bar carries, which the slice makes unnecessary by
            // construction — `end - base` is exactly `self.re.len()`.
            let window = &signal[base as usize..end as usize];
            for ((&s, &wr), &wi) in window.iter().zip(self.re.iter()).zip(self.im.iter()) {
                acc_re += (s * wr) as f64;
                acc_im += (s * wi) as f64;
            }
            return (acc_re, acc_im, 1.0);
        }

        // The surviving taps, as three slices of the same length, for the
        // reason the interior branch takes one: same order, same expressions,
        // same sums, without a bounds check per tap.
        let mut env_sum = 0.0f64;
        let (k0, k1) = ((lo - base) as usize, (hi - base) as usize);
        let window = &signal[lo as usize..hi as usize];
        for (((&s, &wr), &wi), &g) in window
            .iter()
            .zip(&self.re[k0..k1])
            .zip(&self.im[k0..k1])
            .zip(&self.env[k0..k1])
        {
            acc_re += (s * wr) as f64;
            acc_im += (s * wi) as f64;
            env_sum += g as f64;
        }
        (acc_re, acc_im, env_sum)
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

    /// One member's magnitude columns for whatever the run is analysing.
    ///
    /// Mono convolves once. **Joint convolves twice, never three times**, and
    /// derives the mix from the two complex columns before any magnitude is
    /// taken. The complex columns exist only inside this call, so what a run
    /// holds at once is bounded by the number of workers rather than by the
    /// track length times the wavelet count.
    ///
    /// The route each channel takes is chosen the same way it always was, and
    /// the two channels take the same one — a shared block set that is present
    /// for one and missing for the other would put the pair through different
    /// arithmetic, and the mix is a difference of two channels.
    fn member_cols(
        &self, ch: &Chans, hop: usize, frames: usize, shared: &SharedSet,
        work: &WorkCells, seams: &Seams,
    ) -> Cols {
        let k = self.re.len();
        let one = |sig: &[f32], blocks: &Blocks| -> Vec<C32> {
            work.column(1);
            blocks
                .get(&fft_block_len(k))
                .filter(|_| fft_is_cheaper(k, hop))
                .and_then(|b| self.complex_via_shared(sig, b, hop, frames))
                .unwrap_or_else(|| self.complex(sig, hop, frames))
        };
        match ch.b {
            None => Cols { mix: norms(one(ch.a, &shared.a)), ..Default::default() },
            Some(right) => {
                let l = one(ch.a, &shared.a);
                let r = one(right, &shared.b);
                let mix = mix_norms(&l, &r);
                if let Some(mean) = ch.mean.filter(|_| seams.redundant_mix_transform) {
                    // The violation this contract exists to forbid: a third
                    // convolution, of the mean signal. Through an empty block
                    // set, because the shared transforms belong to *left* and
                    // reusing them here would convolve the mean signal against
                    // the wrong signal's blocks and produce a number that means
                    // nothing.
                    let redundant = norms(one(mean, &Blocks::new()));
                    // Consumed and compared: a mean-signal analysis and the
                    // derived mix are the same quantity, so this both proves
                    // the extra work happened and re-checks the derivation.
                    let dev = redundant
                        .iter()
                        .zip(&mix)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0f32, f32::max);
                    work.redundant(dev);
                }
                Cols { mix, left: norms(l), right: norms(r) }
            }
        }
    }

    /// [`Morlet::magnitudes`], kept complex, by whichever route is cheaper.
    fn complex(&self, signal: &[f32], hop: usize, frames: usize) -> Vec<C32> {
        if fft_is_cheaper(self.re.len(), hop)
            && let Some(v) = self.complex_via_fft(signal, hop, frames)
        {
            return v;
        }
        (0..frames).map(|fi| self.response_c(signal, (fi * hop) as isize)).collect()
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
        Some(norms(self.complex_via_shared(signal, blocks, hop, frames)?))
    }

    /// The shared-signal route, kept complex. See [`Morlet::response_c`] for why
    /// the combination has to happen here and not after the magnitude.
    ///
    /// The interior scaling `2/n` is an exact power of two, so applying it to
    /// each component and taking the norm afterwards gives bit-for-bit the
    /// magnitude this route produced before — the mono path is unchanged. Edge
    /// frames, which divide by a surviving envelope sum, can differ in the last
    /// place; they are the few frames whose window hangs off the signal.
    fn complex_via_shared(
        &self, signal: &[f32], blocks: &SignalBlocks, hop: usize, frames: usize,
    ) -> Option<Vec<C32>> {
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

        let mut out = vec![C32 { re: 0.0, im: 0.0 }; frames];
        for (fi, slot) in out.iter_mut().enumerate() {
            if fi < first || fi > last {
                *slot = self.response_c(signal, (fi * hop) as isize);
            }
        }

        let scale = 1.0 / n as f32;
        let mut buf = vec![Complex { re: 0.0f32, im: 0.0f32 }; n];
        // `process` allocates a fresh scratch buffer on every call, which here
        // is once per block. One buffer serves the whole loop instead: rustfft
        // documents its contents as garbage between calls, and the identity
        // tests hold a reused buffer to the bits of a fresh one.
        let mut scratch = vec![Complex { re: 0.0f32, im: 0.0f32 }; inv.get_inplace_scratch_len()];
        for b in 0..blocks.blocks() {
            let base = b * blocks.stride;
            if base >= n_sig { break; }
            // One pass over the block, not two. `Complex::mul_assign` is
            // `*self = *self * rhs`, so a copy followed by `*x *= *h` and a
            // direct `*x = *s * *h` are the same product of the same two
            // values; what goes is a whole-block copy and the pass that read it
            // back.
            for ((x, s), h) in buf.iter_mut().zip(blocks.block(b)).zip(kernel.iter()) {
                *x = *s * *h;
            }
            inv.process_with_scratch(&mut buf, &mut scratch);

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
                    *slot = C32 { re: 2.0 * scale * c.re, im: 2.0 * scale * c.im };
                }
            }
        }
        Some(out)
    }

    fn magnitudes_via_fft(&self, signal: &[f32], hop: usize, frames: usize) -> Option<Vec<f32>> {
        Some(norms(self.complex_via_fft(signal, hop, frames)?))
    }

    /// The own-transform overlap-save route, kept complex. As
    /// [`Morlet::complex_via_shared`].
    fn complex_via_fft(&self, signal: &[f32], hop: usize, frames: usize) -> Option<Vec<C32>> {
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

        let mut out = vec![C32 { re: 0.0, im: 0.0 }; frames];
        // Edge frames, direct.
        for (fi, slot) in out.iter_mut().enumerate() {
            if fi < first || fi > last {
                *slot = self.response_c(signal, (fi * hop) as isize);
            }
        }

        let scale = 1.0 / n as f32;
        let mut buf = vec![Complex { re: 0.0f32, im: 0.0f32 }; n];
        // As in `complex_via_shared`: one scratch for every transform in the
        // loop, sized for whichever direction needs more.
        let scratch_len = fwd.get_inplace_scratch_len().max(inv.get_inplace_scratch_len());
        let mut scratch = vec![Complex { re: 0.0f32, im: 0.0f32 }; scratch_len];
        let mut base = 0usize;
        while base < n_sig {
            // The same values, as one bounded copy and one zero fill rather
            // than a checked lookup per element. `base < n_sig` holds at the
            // top of the loop, so `avail` is exactly where the lookup would
            // start returning `None`, and the tail is exactly the zero padding
            // overlap-save requires.
            let avail = (n_sig - base).min(n);
            for (slot, &x) in buf[..avail].iter_mut().zip(&signal[base..base + avail]) {
                *slot = Complex { re: x, im: 0.0 };
            }
            for slot in buf[avail..].iter_mut() {
                *slot = Complex { re: 0.0, im: 0.0 };
            }
            fwd.process_with_scratch(&mut buf, &mut scratch);
            for (b, h) in buf.iter_mut().zip(kernel.iter()) { *b *= *h; }
            inv.process_with_scratch(&mut buf, &mut scratch);

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
                    *slot = C32 { re: 2.0 * scale * c.re, im: 2.0 * scale * c.im };
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
pub(super) const GPU_MIN_BLOCK: usize = 1 << 17;

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

/// Add one member's contribution to a running log-space accumulator.
fn fold_logs(acc: &mut [f32], col: &[f32], weight: f32) {
    for (a, m) in acc.iter_mut().zip(col) {
        *a += weight * m.max(MAG_FLOOR).ln();
    }
}

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
        // One scratch for every block, as in `Morlet::complex_via_shared`.
        let mut scratch = vec![Complex { re: 0.0f32, im: 0.0 }; fwd.get_inplace_scratch_len()];
        for b in 0..blocks {
            let base = b * stride;
            let dst = &mut data[b * n..(b + 1) * n];
            // As in `Morlet::complex_via_fft`: a bounded copy and a zero fill,
            // the same values a checked lookup per element produced.
            let avail = signal.len().saturating_sub(base).min(n);
            for (slot, &x) in dst[..avail].iter_mut().zip(&signal[base..base + avail]) {
                *slot = Complex { re: x, im: 0.0 };
            }
            for slot in dst[avail..].iter_mut() {
                *slot = Complex { re: 0.0, im: 0.0 };
            }
            fwd.process_with_scratch(dst, &mut scratch);
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

/// What one run analyses.
///
/// The distinction is not "one signal or two". It is **where the mix comes
/// from**: analysed in its own right, or derived from the two channels at the
/// only point where deriving it is correct.
pub enum Input<'a> {
    /// One signal, analysed as itself.
    Mono(&'a [f32]),
    /// A stereo pair. The mix is derived from the two channels' *complex*
    /// responses, member by member, before any magnitude is taken:
    ///
    /// ```text
    /// M_j = (L_j + R_j) / 2
    /// ```
    ///
    /// This is exactly the mix, not an approximation of it: the transform is
    /// linear, and `(l + r)/2` is the signal the mono route would have been
    /// handed. Two convolution sets, not three — and never the average of two
    /// magnitudes, which is a different quantity that reads two anti-phase
    /// channels as loud when their sum is silent.
    Joint { left: &'a [f32], right: &'a [f32] },
}

impl<'a> Input<'a> {
    /// The signal every length, frame count and plan is derived from.
    fn lead(&self) -> &'a [f32] {
        match *self {
            Input::Mono(s) => s,
            Input::Joint { left, .. } => left,
        }
    }

    fn chans(&self) -> Chans<'a> {
        match *self {
            Input::Mono(s) => Chans { a: s, b: None, mean: None },
            // Trimmed to the shorter, so a frame index means the same instant
            // in both channels and the mix is never a combination of two
            // different moments. The producer already pairs them exactly; this
            // makes it a property of the type rather than a promise.
            Input::Joint { left, right } => {
                let m = left.len().min(right.len());
                Chans { a: &left[..m], b: Some(&right[..m]), mean: None }
            }
        }
    }

    pub fn is_joint(&self) -> bool {
        matches!(self, Input::Joint { .. })
    }
}

/// The signals a leaf convolves: one, or two that a mix is derived from.
#[derive(Clone, Copy)]
struct Chans<'a> {
    a: &'a [f32],
    b: Option<&'a [f32]>,
    /// `(a + b)/2`, materialised **only** when the redundant-mix seam is on.
    ///
    /// Production never allocates it. It exists so the deliberate extra
    /// operation is a real convolution of a real third signal, through the same
    /// counted boundary as the two real channels — the first version of this
    /// seam incremented a counter and computed one sample, which made the
    /// mutation evidence a fabrication.
    mean: Option<&'a [f32]>,
}

impl Chans<'_> {
    fn joint(&self) -> bool { self.b.is_some() }
}

/// Forward-transformed signal blocks for each channel of the run.
///
/// Two maps rather than one of pairs, because a size can legitimately be
/// present for one channel and absent for the other: the budget is spent
/// largest-first and the second channel doubles the bill.
#[derive(Default)]
struct SharedSet {
    a: Blocks,
    b: Blocks,
}

type Blocks = std::collections::HashMap<usize, std::sync::Arc<SignalBlocks>>;

/// One bar's finished columns. `left`/`right` are empty unless the run was joint.
#[derive(Clone, Default)]
struct Cols {
    mix: Vec<f32>,
    left: Vec<f32>,
    right: Vec<f32>,
}

impl Cols {
    fn zeroed(frames: usize, joint: bool) -> Self {
        let side = || if joint { vec![0.0f32; frames] } else { Vec::new() };
        Cols { mix: vec![0.0f32; frames], left: side(), right: side() }
    }
}

/// How often cancellation is re-checked inside one bar.
///
/// The lowest bars run for tens of seconds on their own, so a per-bar check
/// alone would leave Abort feeling ignored. Module-scope so the test that
/// exercises the in-bar check reads the production value rather than repeating
/// it.
const CANCEL_CHECK_FRAMES: usize = 512;

/// Move finished columns into their slots.
///
/// The single place where a worker's result becomes the run's result. A `Cols`
/// is one to three `Vec<f32>` of `frames` each, so copying here rather than
/// moving would hold two of every column at once for the length of the loop.
/// All three sweeps go through this, so the test that checks for a copy checks
/// the production path rather than a reimplementation of it in the test file.
fn install_columns(columns: &mut [Option<Cols>], done: impl IntoIterator<Item = (usize, Cols)>) {
    for (bar, col) in done {
        columns[bar] = Some(col);
    }
}

/// Transpose one plane of finished columns into frames × bars, releasing each
/// column as it is consumed.
///
/// The columns and the output hold the same numbers in a different order, so
/// the run does not need both. Transposing all three planes with every column
/// still alive held **six** copies of the largest structure in the run at once
/// — three bars × frames column planes and three frames × bars row planes. One
/// plane at a time, taking each column as it is copied, holds four: the two
/// column planes not yet consumed, the one being consumed, and the output.
///
/// `take` and not `clear`: clearing sets the length to zero and keeps the
/// capacity, which is the entire cost. The taken `Vec` is dropped at the end of
/// each iteration and its allocation goes back to the allocator there.
///
/// A column shorter than `frames` fills the rows it has and leaves the rest at
/// zero, exactly as before; the run is abandoned before this point if any
/// column is missing outright.
fn transpose_plane(
    columns: &mut [Option<Cols>],
    frames: usize,
    n_bars: usize,
    pick: fn(&mut Cols) -> &mut Vec<f32>,
) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0.0f32; n_bars]; frames];
    for (bar, slot) in columns.iter_mut().enumerate() {
        let col = std::mem::take(pick(slot.as_mut().expect("every column is present")));
        for (fi, &v) in col.iter().enumerate() {
            out[fi][bar] = v;
        }
    }
    out
}

/// One bar of CPU work, as little as a worker needs to do it.
///
/// Deliberately three scalars: the kernels are built inside the worker that
/// runs the bar, because a superlet's tables are hundreds of kilobytes at the
/// bottom of the range and building them all up front would hold every bar's
/// kernels at once to save nothing.
#[derive(Clone, Copy)]
struct BarJob {
    bar: usize,
    freq: f32,
    q: f32,
}

/// The scheduling shape both CPU bar sweeps use.
///
/// Rayon sizes a leaf from the *length of the iterator*, halving the split
/// budget on every job that is not stolen. Sweeping the 1024-slot column array
/// and filtering inside it therefore produced leaves of ~128 slots whatever the
/// work in them, and the direct bars — a contiguous handful at the top of the
/// range — all landed in one leaf, which then ran on whichever thread took it.
/// Iterating a compact list of jobs instead makes the length the *work* count,
/// and `with_max_len(1)` sets the splitter's minimum split count to that length
/// (`LengthSplitter::new`), so a leaf holds one job.
///
/// That bounds granularity, and nothing more: it does not promise that six
/// workers are busy, does not guarantee stealing, and is not a deadline. It only
/// removes the case where a whole phase is stuck inside one leaf.
fn bar_jobs(jobs: &[BarJob]) -> impl rayon::iter::IndexedParallelIterator<Item = &BarJob> {
    use rayon::prelude::*;
    jobs.par_iter().with_max_len(1)
}

/// What a run actually dispatched, in units that are each homogeneous.
///
/// **Deliberately not one number.** The routes do differently shaped work — the
/// direct route evaluates one frame at a time, the transform routes produce a
/// whole column per call — and summing them gave a total dominated by whichever
/// route happened to carry the most bars. The earlier single "transform sets"
/// figure was that sum; it is withdrawn, and nothing here should be quoted as a
/// count of physical transforms.
///
/// The counts are also **attempts, not logical work**. A device dispatch that
/// fails sends its whole chunk back to the cores, and both the failed attempt
/// and the recomputation are counted, because the question the counter answers
/// is what the machine did.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct WorkCount {
    /// Whole-column convolutions: one per member, per bar, per channel, per
    /// attempt. Covers the CPU transform routes and each device dispatch.
    pub column_convolutions: u64,
    /// Single-frame direct responses: one per member, per frame, per channel.
    /// The short-kernel bars, and the edge frames a device dispatch leaves for
    /// the host to fill.
    pub frame_responses: u64,
    /// Windowed frame transforms: one per frame, per channel. The FFT bar
    /// mappings' unit. The superlet routes never produce these, and the FFT
    /// producer never produces the other two — so each producer's contract is
    /// checked in its own unit and the sum is never taken.
    pub frame_transforms: u64,
    /// Device dispatches that returned an error. Each one costs its chunk a
    /// recomputation on the cores, so it is reported beside the retries it
    /// causes rather than hidden inside them.
    pub device_failures: u64,
}

impl WorkCount {
    fn snapshot(c: &WorkCells) -> Self {
        use std::sync::atomic::Ordering as O;
        WorkCount {
            column_convolutions: c.columns.load(O::Relaxed),
            frame_responses: c.frames.load(O::Relaxed),
            frame_transforms: 0,
            device_failures: c.device_failures.load(O::Relaxed),
        }
    }
}

/// The live counters behind [`WorkCount`], owned by one run.
///
/// Per invocation, never a static: the suite runs analyses in parallel, and a
/// process-global counter reported whichever run finished last — which is how
/// a joint run once measured as doing a *third* of a mono run's work.
#[derive(Default)]
struct WorkCells {
    columns: std::sync::atomic::AtomicU64,
    frames: std::sync::atomic::AtomicU64,
    device_failures: std::sync::atomic::AtomicU64,
    /// Worst deviation between the redundantly computed mean-signal response
    /// and the mix derived from the two channels, as `f32` bits.
    ///
    /// Recorded so the extra operation's *result* is consumed and checked. An
    /// increment alone proves nothing: it is the same evidence whether a
    /// convolution ran or a counter was bumped.
    redundant_dev: std::sync::atomic::AtomicU32,
    redundant_seen: std::sync::atomic::AtomicBool,
}

impl WorkCells {
    fn column(&self, n: u64) {
        self.columns.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }
    fn frame(&self, n: u64) {
        self.frames.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }
    fn device_failure(&self) {
        self.device_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// The running total, for a caller that wants a delta.
    ///
    /// This counter is cumulative over a whole analysis, so it cannot answer
    /// "did *this group* fall back?" on its own. A group reads it before and
    /// after and compares — otherwise every group after the first failure would
    /// look like a failure too, and one flaky dispatch would invalidate the
    /// rest of the run.
    fn device_failures_now(&self) -> u64 {
        self.device_failures.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Record the largest deviation seen so far between a redundantly computed
    /// mix and the derived one.
    fn redundant(&self, dev: f32) {
        use std::sync::atomic::Ordering as O;
        self.redundant_seen.store(true, O::Relaxed);
        let bits = dev.to_bits();
        let mut cur = self.redundant_dev.load(O::Relaxed);
        while dev > f32::from_bits(cur) {
            match self.redundant_dev.compare_exchange_weak(cur, bits, O::Relaxed, O::Relaxed) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }

    fn redundant_result(&self) -> Option<f32> {
        use std::sync::atomic::Ordering as O;
        self.redundant_seen
            .load(O::Relaxed)
            .then(|| f32::from_bits(self.redundant_dev.load(O::Relaxed)))
    }
}

/// Knobs the driver exposes to its own tests.
///
/// Not configuration. They exist so a test can put the driver into a state a
/// real machine reaches rarely and unpredictably — a memory budget too small
/// for a block size, a device that declines partway through a run — and then
/// check what the *production* orchestration does about it, rather than
/// checking a reimplementation of its decision logic.
#[derive(Clone, Copy)]
struct Seams<'a> {
    /// Ceiling on one group's shared signal transforms, in bytes.
    shared_budget: usize,
    /// Called once per device dispatch with `(chunk index, channel index)`.
    /// Returning `true` makes that dispatch fail exactly as a device would.
    dispatch_fault: Option<&'a (dyn Fn(usize, usize) -> bool + Sync)>,
    /// Taps to hold at once while staging a chunk of kernels for the device.
    ///
    /// A test lowers it to force a group into several chunks, which is the only
    /// way to reach the case where one chunk is already staged and waiting to
    /// be assembled when a later dispatch fails.
    kernel_tap_budget: usize,
    /// Convolve the mean signal as well, and throw the result away.
    ///
    /// A deliberate violation of the two-channel contract, so that the test
    /// which checks the contract can be shown to fail when it is broken. A
    /// check nothing can fail is not a check.
    redundant_mix_transform: bool,
}

impl Seams<'_> {
    fn production() -> Self {
        Seams {
            shared_budget: SHARED_BLOCKS_MAX_BYTES,
            kernel_tap_budget: GPU_KERNEL_TAP_BUDGET,
            dispatch_fault: None,
            redundant_mix_transform: false,
        }
    }
}

/// Everything one run produced. `left`/`right` are empty for a mono run.
pub struct Analysed {
    pub mix: Vec<Vec<f32>>,
    pub left: Vec<Vec<f32>>,
    pub right: Vec<Vec<f32>>,
    /// What the run dispatched. See [`WorkCount`].
    pub work: WorkCount,
    /// Worst deviation between the redundant mean-signal analysis and the
    /// derived mix, when the redundant-mix seam was on. `None` in production
    /// and in every run that did not ask for it.
    ///
    /// The seam's purpose is to make the work-count contract fail; this is how
    /// a test knows the extra work was *performed* rather than merely counted.
    pub redundant_mix_deviation: Option<f32>,
}

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

    /// One frame, for whatever the run is analysing — the direct route's leaf.
    ///
    /// The mix is combined per member and per frame, before the floor, the log
    /// and the mean, exactly as on the transform routes. Doing it once at the
    /// end of the aggregation would be averaging two geometric means, which is
    /// neither the mix nor anything else.
    fn frame(
        &self, ch: &Chans, centre: isize, work: &WorkCells, seams: &Seams,
    ) -> (f32, f32, f32) {
        // Counting is bound to the call, not written beside it: removing a
        // response removes its count, and no increment can outlive the work it
        // is supposed to describe.
        let resp = |w: &Morlet, sig: &[f32]| -> C32 {
            work.frame(1);
            w.response_c(sig, centre)
        };
        let mut acc = (0.0f32, 0.0f32, 0.0f32);
        // Only ever accumulated when the redundant-mix seam is on.
        let mut acc_redundant = 0.0f32;
        for (w, weight) in self.wavelets.iter().zip(self.weights.iter()) {
            match ch.b {
                None => {
                    work.frame(1);
                    acc.0 += weight * w.response(ch.a, centre).max(MAG_FLOOR).ln();
                }
                Some(right) => {
                    let l = resp(w, ch.a);
                    let r = resp(w, right);
                    let m = C32 { re: 0.5 * (l.re + r.re), im: 0.5 * (l.im + r.im) };
                    let mag = |c: C32| (c.re * c.re + c.im * c.im).sqrt().max(MAG_FLOOR).ln();
                    if let Some(mean) = ch.mean.filter(|_| seams.redundant_mix_transform) {
                        // A real third response, of a real third signal,
                        // through the same counted boundary as the other two.
                        acc_redundant += weight * mag(resp(w, mean));
                    }
                    acc.0 += weight * mag(m);
                    acc.1 += weight * mag(l);
                    acc.2 += weight * mag(r);
                }
            }
        }
        let inv = 1.0 / self.total_weight;
        let out = ((acc.0 * inv).exp(), (acc.1 * inv).exp(), (acc.2 * inv).exp());
        if ch.mean.is_some() && seams.redundant_mix_transform && ch.b.is_some() {
            // Consumed and checked, so the count cannot be the only evidence.
            work.redundant(((acc_redundant * inv).exp() - out.0).abs());
        }
        out
    }

    /// Every frame at once, for whatever the run is analysing.
    ///
    /// Members are folded in one at a time, so only one member's columns are
    /// alive at once however many wavelets a bar has — the same bound
    /// [`Superlet::responses_with`] kept, now over three outputs instead of one.
    fn columns(
        &self, ch: &Chans, hop: usize, frames: usize, shared: &SharedSet,
        work: &WorkCells, seams: &Seams,
    ) -> Cols {
        let joint = ch.joint();
        let mut acc = Cols::zeroed(frames, joint);
        for (w, weight) in self.wavelets.iter().zip(self.weights.iter()) {
            let c = w.member_cols(ch, hop, frames, shared, work, seams);
            fold_logs(&mut acc.mix, &c.mix, *weight);
            if joint {
                fold_logs(&mut acc.left, &c.left, *weight);
                fold_logs(&mut acc.right, &c.right, *weight);
            }
        }
        let inv = 1.0 / self.total_weight;
        let finish = |v: Vec<f32>| v.into_iter().map(|a| (a * inv).exp()).collect();
        Cols {
            mix: finish(acc.mix),
            left: if joint { finish(acc.left) } else { Vec::new() },
            right: if joint { finish(acc.right) } else { Vec::new() },
        }
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
        Input::Mono(signal), sample_rate, n_bars, min_freq, max_freq, hop, cfg,
        should_continue, on_bar_done, route_to_device, Seams::production(),
    )
    .mix
}

/// A stereo analysis: left, right, and the mix **derived** from them.
///
/// Not three analyses. The two channels are convolved and the mix is taken at
/// `M_j = (L_j + R_j)/2` on the complex per-member responses, before magnitude,
/// floor, log and the weighted geometric mean — see [`Input::Joint`]. The result
/// is the mix, not an approximation of it, because the transform is linear and
/// `(l + r)/2` is the signal a mono run would have been given.
///
/// Every route takes the pair: direct, own-transform overlap-save,
/// shared-transform, and the device. None of them falls back to analysing a
/// third signal.
#[allow(clippy::too_many_arguments)]
pub fn analyze_input_with_progress(
    input: Input<'_>,
    sample_rate: u32,
    n_bars: usize,
    min_freq: f32,
    max_freq: f32,
    hop: usize,
    cfg: &AsltConfig,
    should_continue: &(dyn Fn() -> bool + Sync),
    on_bar_done: &(dyn Fn(usize) + Sync),
) -> Analysed {
    analyze_routed(
        input, sample_rate, n_bars, min_freq, max_freq,
        hop, cfg, should_continue, on_bar_done, route_to_device, Seams::production(),
    )
}

/// [`analyze_with_progress`], with the device-routing decision supplied.
///
/// Production passes [`route_to_device`]; tests pass whatever they are actually
/// asserting about, so that no two tests can steer each other by writing to a
/// shared threshold.
#[allow(clippy::too_many_arguments)]
fn analyze_routed(
    input: Input<'_>,
    sample_rate: u32,
    n_bars: usize,
    min_freq: f32,
    max_freq: f32,
    hop: usize,
    cfg: &AsltConfig,
    should_continue: &(dyn Fn() -> bool + Sync),
    on_bar_done: &(dyn Fn(usize) + Sync),
    route: RouteDecision,
    seams: Seams<'_>,
) -> Analysed {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    let empty = || Analysed {
        mix: Vec::new(),
        left: Vec::new(),
        right: Vec::new(),
        work: WorkCount::default(),
        redundant_mix_deviation: None,
    };
    let mut chans = input.chans();
    let joint = chans.joint();
    // Only the seam allocates this, and only a test sets the seam.
    let mean_signal: Option<Vec<f32>> = (seams.redundant_mix_transform && joint)
        .then(|| {
            let b = chans.b.expect("joint");
            chans.a.iter().zip(b).map(|(x, y)| 0.5 * (x + y)).collect()
        });
    chans.mean = mean_signal.as_deref();
    // Every length, plan and frame count comes from one signal. A joint run has
    // already been trimmed to the shorter of the two, so the two channels are
    // the same length by construction and a frame index means the same instant
    // in both.
    let signal = chans.a;
    let frames = frame_count(signal.len(), hop);
    if frames == 0 || n_bars == 0 { return empty(); }
    stats_reset();
    let work = WorkCells::default();
    let t_run = std::time::Instant::now();

    let sr = sample_rate as f32;
    let nyquist = sr * 0.5;
    // Per bar, not one number for the whole spectrum. Under `Log` this returns
    // the same closed form it always did, identically for every bar; under
    // `Erb` the bars are not evenly spaced in log frequency, so the resolution
    // the grid can draw varies along it.
    let grid_at = |bar: usize| cfg.scale.grid_q_at(bar, n_bars, min_freq, max_freq);
    let cancelled = AtomicBool::new(false);

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

    let mut columns: Vec<Option<Cols>> = vec![None; n_bars];
    for bar in above_nyquist {
        columns[bar] = Some(Cols::zeroed(frames, joint));
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
        // A joint run needs one set per channel, so a size costs twice as much
        // and the same budget holds half as many sizes — which is a slowdown,
        // not a wrong answer: a size that does not fit falls back to the
        // per-kernel route for both channels alike.
        let per_channel =
            2 * signal.len() * std::mem::size_of::<rustfft::num_complex::Complex<f32>>();
        if joint { 2 * per_channel } else { per_channel }
    };

    for (_key, idxs) in groups.iter().rev() {
        if cancelled.load(Ordering::Relaxed) { break; }

        let mut wanted: Vec<usize> = idxs.iter()
            .flat_map(|&i| plans[i].sizes.iter().copied())
            .collect();
        wanted.sort_unstable_by(|a, b| b.cmp(a)); // largest first
        wanted.dedup();

        let mut shared = SharedSet::default();
        let mut spent = 0usize;
        for n in wanted {
            let cost = block_bytes(n);
            if spent + cost > seams.shared_budget { break; }
            spent += cost;
            // Both channels or neither. A size present for one and missing for
            // the other would put the pair through different routes, and the
            // mix is a combination of the two — the routes are equivalent to
            // within the transform's own tolerance, not bit-identical.
            shared.a.insert(n, std::sync::Arc::new(SignalBlocks::build(signal, n)));
            if let Some(right) = chans.b {
                shared.b.insert(n, std::sync::Arc::new(SignalBlocks::build(right, n)));
            }
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
        // One decision belongs to one group. The scope retires it however this
        // iteration ends — recorded, cancelled, returned early from, or
        // unwound — so the next group cannot inherit it and record its own time
        // under this group's block size.
        let group_scope = super::gpu_calib::begin_group();
        let wanted_device = route(group_n);
        let gpu = wanted_device
            .then(super::gpu::GpuFft::shared)
            .flatten();
        let t_group = std::time::Instant::now();
        // Cumulative over the analysis, so the per-group question is a delta.
        let failures_before = work.device_failures_now();

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
                    if end > at && taps + cost > seams.kernel_tap_budget { break; }
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
            /// One complex column per kernel, per channel. `b` is empty on a
            /// mono run. The device returns complex now, because the mix has to
            /// be taken before the magnitude and the shader is one of the four
            /// places that magnitude used to be taken.
            cols_a: Vec<Vec<super::gpu::C32>>,
            cols_b: Vec<Vec<super::gpu::C32>>,
        }

        // Turn a finished chunk into finished bars. Pulled out of the loop
        // because it now runs one iteration behind the dispatch that produced
        // it, and once more at the end to drain the last one.
        let assemble = |p: Staged| -> Vec<(usize, Cols)> {
            let t_fill = std::time::Instant::now();
            let Staged { sls, owner, cols_a, cols_b } = p;
            let blank = || -> Vec<Vec<Option<Vec<super::gpu::C32>>>> {
                sls.iter().map(|(_, sl)| vec![None; sl.wavelets.len()]).collect()
            };
            let mut got_a = blank();
            let mut got_b = blank();
            for ((ci, wi), col) in owner.iter().zip(cols_a) {
                got_a[*ci][*wi] = Some(col);
            }
            for ((ci, wi), col) in owner.iter().zip(cols_b) {
                got_b[*ci][*wi] = Some(col);
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
            // One channel of one member, wherever it came from.
            let channel = |w: &Morlet,
                           dev: &Option<Vec<super::gpu::C32>>,
                           sig: &[f32],
                           blocks: &Blocks| -> Vec<C32> {
                match dev {
                    // Frames the device marked NaN are edge frames, whose window
                    // runs past the signal and renormalises over the taps that
                    // survive — a convolution cannot express that, so they come
                    // from the direct path exactly as on the CPU route. The
                    // sentinel is a NaN and not the old `-1.0` because every
                    // real number is a legal component of a complex response.
                    // Already counted where it was dispatched. The edge
                    // frames are not: they are host work the device left
                    // undone, and they are per-frame, not a column.
                    Some(col) => col.iter().enumerate()
                        .map(|(fi, v)| if v[0].is_nan() {
                            work.frame(1);
                            w.response_c(sig, (fi * hop) as isize)
                        } else {
                            C32 { re: v[0], im: v[1] }
                        })
                        .collect(),
                    // The wavelets the device did not take are the smaller
                    // members of the same bar, and they get the group's shared
                    // signal transform exactly as they would on the CPU route —
                    // calling the bare per-kernel route here instead cost a
                    // full-signal forward FFT per wavelet.
                    None => {
                        work.column(1);
                        let k = w.taps();
                        blocks
                            .get(&fft_block_len(k))
                            .filter(|_| fft_is_cheaper(k, hop))
                            .and_then(|b| w.complex_via_shared(sig, b, hop, frames))
                            .unwrap_or_else(|| w.complex(sig, hop, frames))
                    }
                }
            };

            let mags: Vec<Cols> = pairs.par_iter()
                .map(|&(ci, wi)| {
                    let w = &sls[ci].1.wavelets[wi];
                    let l = channel(w, &got_a[ci][wi], chans.a, &shared.a);
                    match chans.b {
                        None => Cols { mix: norms(l), ..Default::default() },
                        Some(right) => {
                            let r = channel(w, &got_b[ci][wi], right, &shared.b);
                            let mix = mix_norms(&l, &r);
                            Cols { mix, left: norms(l), right: norms(r) }
                        }
                    }
                })
                .collect();

            let mut out = Vec::with_capacity(sls.len());
            let mut taken = 0usize;
            for (bar, sl) in sls.iter() {
                let n = sl.wavelets.len();
                let members = &mags[taken..taken + n];
                let pick = |f: fn(&Cols) -> &Vec<f32>| -> Vec<f32> {
                    let cols: Vec<Vec<f32>> = members.iter().map(|c| f(c).clone()).collect();
                    combine_logs(&cols, &sl.weights, sl.total_weight, frames)
                };
                let col = Cols {
                    mix: pick(|c| &c.mix),
                    left: if joint { pick(|c| &c.left) } else { Vec::new() },
                    right: if joint { pick(|c| &c.right) } else { Vec::new() },
                };
                taken += n;
                on_bar_done(*bar);
                out.push((*bar, col));
            }
            STATS.fill_ns.fetch_add(
                t_fill.elapsed().as_nanos() as u64, Ordering::Relaxed);
            out
        };

        // From here the exploration has really happened: a cancellation after
        // this point spends the attempt rather than refunding it, which is what
        // stops repeated start-and-cancel from buying unlimited exploration.
        group_scope.work_started();

        let (gpu_cols, cpu_cols) = rayon::join(
            || -> Vec<(usize, Cols)> {
                let Some(g) = gpu else { return Vec::new() };
                let mut out: Vec<(usize, Cols)> = Vec::new();
                // Once for the whole block size, never per chunk — and not at
                // all until a chunk is actually going to use it. Preparing
                // eagerly cost an upload and a full-signal transform for every
                // large group whose chunks then turned out to be too small to
                // be worth the device, which was most of Extreme's.
                // One per channel. A joint run stages both; the second is
                // the same size as the first, and if either fails to stage the
                // chunk falls through to the cores rather than analysing one
                // channel on the device and the other off it.
                let mut prepared: Option<(super::gpu::GpuSignal, Option<super::gpu::GpuSignal>)> =
                    None;
                // The chunk whose transforms are finished but whose columns are
                // not yet assembled — always exactly one behind the dispatch.
                let mut staged: Option<Staged> = None;

                for (chunk_i, &(at, end)) in chunks.iter().enumerate() {
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
                        prepared = g.prepare_signal(signal, group_n, stride).ok()
                            .and_then(|a| match chans.b {
                                None => Some((a, None)),
                                Some(right) => g
                                    .prepare_signal(right, group_n, stride)
                                    .ok()
                                    .map(|b| (a, Some(b))),
                            });
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
                            // Two dispatches on a joint run, one per channel,
                            // over the same kernels. That is the two-convolution
                            // contract on this route: never a third for the mix.
                            // Counted here, at the dispatch, and once per
                            // channel per attempt — not later in assembly,
                            // where a failed dispatch would never be seen.
                            let send = |sg: &super::gpu::GpuSignal, ci: usize| {
                                work.column(kernels.len() as u64);
                                if seams.dispatch_fault.is_some_and(|f| f(chunk_i, ci)) {
                                    return Err(format!(
                                        "injected device failure, chunk {chunk_i} channel {ci}"
                                    ));
                                }
                                g.convolve_with(sg, signal.len(), hop, frames, &kernels)
                            };
                            let r = match &prepared {
                                Some((a, b)) => send(a, 0).and_then(|ca| match b {
                                    None => Ok((ca, Vec::new())),
                                    Some(b) => send(b, 1).map(|cb| (ca, cb)),
                                }),
                                None => Err("signal could not be staged".into()),
                            };
                            if r.is_err() {
                                work.device_failure();
                            }
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
                        Ok((cols_a, cols_b))
                            if cols_a.len() == n_kernels
                                && (!joint || cols_b.len() == n_kernels) =>
                        {
                            staged = Some(Staged { sls, owner, cols_a, cols_b });
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
            || -> Vec<(usize, Cols)> {
                // Everything the device did not take, across every core, while
                // it works. These are the smaller-kernel bars of the group, so
                // there are usually many more of them than there are GPU bars.
                cpu_bars.par_iter()
                    .filter_map(|&bar| {
                        if cancel_at_bar(bar, &cancelled) { return None; }
                        let p = by_bar[&bar];
                        let sl = Superlet::new(p.freq, p.q, cfg, sr);
                        let col = sl.columns(&chans, hop, frames, &shared, &work, &seams);
                        on_bar_done(bar);
                        Some((bar, col))
                    })
                    .collect()
            },
        );

        install_columns(&mut columns, gpu_cols.into_iter().chain(cpu_cols));

        // Normally empty: only a device that declined mid-run leaves anything
        // here, and then this is the identical computation on the cores.
        //
        // The jobs are built *after* the results above are installed, so the
        // condition is the same one the old sweep applied per slot: a bar of
        // this group whose column is still missing.
        let missing: Vec<BarJob> = (0..n_bars)
            .filter(|bar| members.contains(bar) && columns[*bar].is_none())
            .map(|bar| {
                let p = by_bar[&bar];
                BarJob { bar, freq: p.freq, q: p.q }
            })
            .collect();
        if !missing.is_empty() {
            let recovered: Vec<(usize, Cols)> = bar_jobs(&missing)
                .filter_map(|job| {
                    if cancel_at_bar(job.bar, &cancelled) { return None; }
                    let sl = Superlet::new(job.freq, job.q, cfg, sr);
                    let col = sl.columns(&chans, hop, frames, &shared, &work, &seams);
                    on_bar_done(job.bar);
                    Some((job.bar, col))
                })
                .collect();
            install_columns(&mut columns, recovered);
        }

        // What this group cost, and which route actually paid it. Units are
        // bars × frames so a 20 s track and a 5 minute one are comparable. A
        // cancelled group is not a measurement of anything and is dropped —
        // dropping the decision returns the exploration attempt it reserved.
        //
        // `gpu.is_some()` is not the outcome. It says a device was *selected*;
        // the recovery sweep above runs declined bars on the cores, and a group
        // that went both ways is a mixture whose wall time belongs to neither
        // route. Charging it to the device is how a flaky card teaches this
        // machine that the route it never cleanly ran is the better one.
        if !cancelled.load(Ordering::Relaxed) {
            let outcome = match (wanted_device, gpu.is_some()) {
                (false, _) => super::gpu_calib::Outcome::Cores,
                // Asked and got nothing: the cores did the work, but only after
                // paying for a staging attempt a genuine cores route never pays.
                (true, false) => super::gpu_calib::Outcome::Unavailable,
                (true, true) if work.device_failures_now() > failures_before => {
                    super::gpu_calib::Outcome::PartialFallback
                }
                (true, true) => super::gpu_calib::Outcome::Device,
            };
            super::gpu_calib::record(
                (bars.len() * frames) as f64,
                t_group.elapsed().as_secs_f64(),
                outcome,
            );
        }
        // Explicit, rather than left to the end of the iteration: a cancelled
        // group skips the `record` above entirely, and this is the path that
        // retires its decision instead of leaving it for the next group.
        drop(group_scope);
    }

    // The short-kernel bars, which never touch a transform. They keep the
    // finer-grained cancellation checks: one of these really can run for tens of
    // seconds, where a frequency-domain bar cannot be stopped part-way anyway.
    if !cancelled.load(Ordering::Relaxed) {
        let jobs: Vec<BarJob> = direct_bars.iter()
            .map(|&(bar, freq, q)| BarJob { bar, freq, q })
            .collect();
        let done: Vec<(usize, Cols)> = bar_jobs(&jobs)
            .filter_map(|job| {
                if cancelled.load(Ordering::Relaxed) { return None; }
                let sl = Superlet::new(job.freq, job.q, cfg, sr);
                let mut col = Cols::zeroed(0, joint);
                col.mix.reserve(frames);
                for fi in 0..frames {
                    if fi % CANCEL_CHECK_FRAMES == 0
                        && (cancelled.load(Ordering::Relaxed) || !should_continue())
                    {
                        cancelled.store(true, Ordering::Relaxed);
                        // The bar is abandoned, so its slot stays empty and the
                        // run ends with nothing, exactly as before.
                        return None;
                    }
                    // Both channels and the mix from one pass over the members,
                    // so the direct route pays for two convolutions and not
                    // three, exactly as the transform routes do.
                    let (m, l, r) = sl.frame(&chans, (fi * hop) as isize, &work, &seams);
                    col.mix.push(m);
                    if joint {
                        col.left.push(l);
                        col.right.push(r);
                    }
                }
                on_bar_done(job.bar);
                Some((job.bar, col))
            })
            .collect();
        // Moved, not copied: each finished column goes straight into its slot.
        install_columns(&mut columns, done);
    }

    // Recorded before the cancellation check so an aborted run still reports
    // what it managed — that is exactly when someone is watching the panel.
    STATS.total_ns.store(t_run.elapsed().as_nanos() as u64, Ordering::Relaxed);

    if cancelled.load(Ordering::Relaxed) || columns.iter().any(|c| c.is_none()) {
        return empty();
    }

    // Transpose to frames × bars, the layout the cache and renderer expect.
    // One plane at a time, so the columns are released as they are consumed
    // rather than all being held until the last output is finished.
    let mix = transpose_plane(&mut columns, frames, n_bars, |c| &mut c.mix);
    let mut left = Vec::new();
    let mut right = Vec::new();
    if joint {
        left = transpose_plane(&mut columns, frames, n_bars, |c| &mut c.left);
        right = transpose_plane(&mut columns, frames, n_bars, |c| &mut c.right);
    }
    drop(columns);
    Analysed {
        mix,
        left,
        right,
        work: WorkCount::snapshot(&work),
        redundant_mix_deviation: work.redundant_result(),
    }
}

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

/// Rough whole-machine throughput, used only for the very first ETA before any
/// real bars have finished. Observed at ~7–8 G taps/s on an eight-core release
/// build; deliberately pessimistic so the estimate falls rather than climbs,
/// and replaced by the measured rate within the first few percent.
///
/// **This is a rate of modelled work, not of instructions.** A tap is not a
/// multiply-accumulate the CPU performs — see [`bar_taps_per_frame`] — so
/// dividing a tap count by this figure gives a time, and dividing it by a
/// clock speed gives nothing at all. Neither the figure nor the tap count says
/// how many cores a preset needs.
pub const TAPS_PER_SEC_HINT: f64 = 5.0e9;

/// Wavelet taps each bar costs *per frame*.
///
/// Exposed as a per-bar vector rather than a single total because bar cost spans
/// four orders of magnitude — a 20 Hz bar can be 1000× a 20 kHz one. Counting
/// completed *bars* would make the progress bar crawl and then leap; weighting
/// by taps makes it linear, and makes the ETA honest.
///
/// # A tap is a unit of cost, not an operation
///
/// A tap is one wavelet sample of one member of one superlet: the total kernel
/// length the analysis has to account for. It is deliberately *not* a count of
/// arithmetic the machine performs, and quoting it as multiply-accumulates,
/// FLOPs or a core requirement overstates it by a large and variable factor:
///
/// * long kernels never run the direct inner loop at all. [`fft_is_cheaper`]
///   routes them through overlap-save, which costs on the order of
///   `L log N` per output rather than `L` per output per tap — and those are
///   exactly the bass bars that dominate the total, so the discrepancy is
///   largest where the number is largest.
/// * the frequency-domain route shares one set of signal blocks across every
///   kernel of a block size, so a chunk of bars pays for the forward transform
///   once between them.
/// * the GPU takes the largest blocks entirely, where the relationship between
///   a tap and a device instruction is different again.
///
/// What the number is good for is what it is used for: comparing one
/// configuration against another, and turning a measured taps-per-second into
/// a remaining time. Both are ratios, and the modelling error cancels.
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
        analyze_routed(
            Input::Mono(sig), sr, bars, lo, hi, hop, cfg, &|| true, &|_| {}, route,
            Seams::production(),
        )
        .mix
    }

    /// The joint form of [`analyze_via`], for the tests that hold the derived
    /// mix to an independently analysed one.
    #[allow(clippy::too_many_arguments)]
    fn analyze_joint_via(
        route: RouteDecision,
        left: &[f32],
        right: &[f32],
        sr: u32,
        bars: usize,
        lo: f32,
        hi: f32,
        hop: usize,
        cfg: &AsltConfig,
    ) -> Analysed {
        analyze_joint_seamed(route, Seams::production(), left, right, sr, bars, lo, hi, hop, cfg)
    }

    /// The joint driver with the seams a test needs to reach a state a real
    /// machine reaches rarely. Production orchestration throughout — only the
    /// budget and the device's behaviour are supplied.
    #[allow(clippy::too_many_arguments)]
    fn analyze_joint_seamed(
        route: RouteDecision,
        seams: Seams<'_>,
        left: &[f32],
        right: &[f32],
        sr: u32,
        bars: usize,
        lo: f32,
        hi: f32,
        hop: usize,
        cfg: &AsltConfig,
    ) -> Analysed {
        analyze_routed(
            Input::Joint { left, right }, sr, bars, lo, hi, hop, cfg,
            &|| true, &|_| {}, route, seams,
        )
    }

    // ── The derived mix, against an independently analysed one ──────────────
    //
    // The oracle is a *separate mono analysis of the mean signal*, used only as
    // a test oracle: production never runs it. If the joint path is right, the
    // mix it derives from two complex response sets equals the mix that
    // analysing `(l + r)/2` would have produced, to within float error — and
    // that is the whole claim behind removing the third pass.

    /// Signals whose mixes are interesting for different reasons.
    fn joint_cases(sr: u32, n: usize) -> Vec<(&'static str, Vec<f32>, Vec<f32>)> {
        let tau = std::f32::consts::TAU;
        let t = |i: usize| i as f32 / sr as f32;
        let tone = |f: f32, a: f32, ph: f32| -> Vec<f32> {
            (0..n).map(|i| a * (tau * f * t(i) + ph).sin()).collect()
        };
        let mut seed = 0x1234_5678_9ABC_DEF1u64;
        let mut noise = || -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed >> 40) as f32 / 8_388_608.0) - 1.0
                })
                .collect()
        };
        // A click, so the mix is tested where the signal is not stationary.
        let transient: Vec<f32> = (0..n)
            .map(|i| if i == n / 3 { 0.9 } else { 0.0 })
            .collect();
        let silent = vec![0.0f32; n];

        vec![
            ("in phase", tone(220.0, 0.6, 0.0), tone(220.0, 0.6, 0.0)),
            // The case the whole design turns on: both loud, mix silent.
            ("anti phase", tone(220.0, 0.6, 0.0), tone(220.0, 0.6, std::f32::consts::PI)),
            ("one silent", tone(300.0, 0.7, 0.0), silent.clone()),
            ("independent", tone(180.0, 0.5, 0.0), tone(700.0, 0.5, 1.1)),
            ("noise", noise(), noise()),
            ("transient", transient.clone(), tone(500.0, 0.4, 0.0)),
            // Both channels at the floor: the log floor and the geometric mean
            // must not turn silence into something.
            ("silence", silent.clone(), silent),
        ]
    }

    /// The mean signal, in f32, exactly as a mono analysis would be handed it.
    fn mean_of(l: &[f32], r: &[f32]) -> Vec<f32> {
        l.iter().zip(r).map(|(a, b)| 0.5 * (a + b)).collect()
    }

    /// Worst difference between two spectrograms, relative to `full_scale`.
    ///
    /// The reference is passed in rather than taken from the oracle, because
    /// the anti-phase case has a numerically silent oracle: both sides compute
    /// something around 1e-8, and dividing by *that* turns agreement in the
    /// eighth decimal place into a 1 % relative error. What the comparison is
    /// actually about is whether the mix is right on the scale of the audio
    /// being analysed, so that is the scale it is measured on.
    fn worst_rel_to(got: &[Vec<f32>], want: &[Vec<f32>], full_scale: f32) -> f32 {
        let peak = full_scale.max(1e-12);
        got.iter()
            .flatten()
            .zip(want.iter().flatten())
            .map(|(a, b)| (a - b).abs() / peak)
            .fold(0.0f32, f32::max)
    }

    fn peak_of(rows: &[Vec<f32>]) -> f32 {
        rows.iter().flatten().cloned().fold(0.0f32, f32::max)
    }

    /// The common case: judged against the oracle's own level.
    fn worst_rel(got: &[Vec<f32>], want: &[Vec<f32>]) -> f32 {
        worst_rel_to(got, want, peak_of(want))
    }

    #[test]
    fn the_derived_mix_matches_an_independently_analysed_one() {
        let sr = 16_000u32;
        let n = (2.0 * sr as f32) as usize;
        let hop = hop_for_fps(sr, 60.0);
        // A range wide enough that both routes are exercised in one run: the
        // low bars are long-kernel and go through the transform, the high bars
        // are short-kernel and go through the direct loop.
        let (lo, hi) = (30.0f32, 6_000.0);
        let bars = 96;
        let cfg = cfg(1.0, 1.2, 4, 0.5);

        for (name, l, r) in joint_cases(sr, n) {
            let joint = analyze_joint_via(never_gpu, &l, &r, sr, bars, lo, hi, hop, &cfg);
            let oracle = analyze_via(never_gpu, &mean_of(&l, &r), sr, bars, lo, hi, hop, &cfg);

            assert_eq!(joint.mix.len(), oracle.len(), "{name}: frame count");
            assert!(!joint.mix.is_empty(), "{name}: nothing produced");

            let ol = analyze_via(never_gpu, &l, sr, bars, lo, hi, hop, &cfg);
            let or = analyze_via(never_gpu, &r, sr, bars, lo, hi, hop, &cfg);
            // Full scale for this case is the loudest of the three spectra, so
            // the anti-phase case — whose oracle is numerically silent — is
            // judged on the scale of the audio rather than on the scale of its
            // own rounding error.
            let fs = peak_of(&oracle).max(peak_of(&ol)).max(peak_of(&or));

            let worst = worst_rel_to(&joint.mix, &oracle, fs);
            println!("  mix vs oracle, {name:<12} worst {worst:.2e} of full scale {fs:.3}");
            assert!(
                worst < 2e-3,
                "{name}: the derived mix differs from an independent analysis by {worst:.3e}"
            );

            // And the channels are themselves, not the mix.
            assert!(worst_rel_to(&joint.left, &ol, fs) < 2e-3, "{name}: left");
            assert!(worst_rel_to(&joint.right, &or, fs) < 2e-3, "{name}: right");
        }
    }

    /// The case an average of magnitudes gets wrong, stated as a number.
    ///
    /// Two channels in opposite polarity sum to silence. A joint run must read
    /// the mix as silent while reading both channels as loud — and it must do
    /// so *without* a third analysis, which is exactly what the derivation
    /// buys.
    #[test]
    fn anti_phase_channels_mix_to_silence_and_not_to_their_average() {
        let sr = 16_000u32;
        let n = (2.0 * sr as f32) as usize;
        let tau = std::f32::consts::TAU;
        let l: Vec<f32> = (0..n)
            .map(|i| 0.7 * (tau * 400.0 * i as f32 / sr as f32).sin())
            .collect();
        let r: Vec<f32> = l.iter().map(|s| -s).collect();

        let hop = hop_for_fps(sr, 60.0);
        let cfg = cfg(1.0, 1.2, 4, 0.5);
        let got = analyze_joint_via(never_gpu, &l, &r, sr, 64, 100.0, 2_000.0, hop, &cfg);

        let mid = got.mix.len() / 2;
        let peak = |row: &Vec<f32>| row.iter().cloned().fold(0.0f32, f32::max);
        let (m, pl, pr) = (peak(&got.mix[mid]), peak(&got.left[mid]), peak(&got.right[mid]));
        println!("  anti-phase: mix {m:.3e}  left {pl:.3e}  right {pr:.3e}");

        assert!(pl > 0.2, "setup: left should be loud, got {pl:.3e}");
        assert!(pr > 0.2, "setup: right should be loud, got {pr:.3e}");
        // The average of the two magnitudes would be `(pl + pr)/2`. The mix is
        // orders of magnitude below it, because it is the magnitude of the sum.
        assert!(
            m < 0.01 * (pl + pr) * 0.5,
            "the mix read {m:.3e} against an average of {:.3e}; something is \
             combining magnitudes rather than responses",
            (pl + pr) * 0.5
        );
    }

    /// A mono run's work count, through the same driver.
    #[allow(clippy::too_many_arguments)]
    fn analyze_mono_work(
        sig: &[f32], sr: u32, bars: usize, lo: f32, hi: f32, hop: usize, cfg: &AsltConfig,
    ) -> WorkCount {
        analyze_routed(
            Input::Mono(sig), sr, bars, lo, hi, hop, cfg, &|| true, &|_| {}, never_gpu,
            Seams::production(),
        )
        .work
    }

    /// The two-channel contract, as a function, so that a test can show it
    /// rejecting a violation as well as accepting the real thing.
    ///
    /// Each unit is compared on its own. Summing them would let an extra
    /// column convolution hide behind a few hundred thousand frame responses,
    /// which is what the earlier single total did.
    fn two_channel_contract(mono: WorkCount, joint: WorkCount) -> Result<(), String> {
        if joint.device_failures != 0 || mono.device_failures != 0 {
            return Err(format!(
                "device failures make physical work counts unequal: mono {}, joint {}",
                mono.device_failures, joint.device_failures
            ));
        }
        for (what, m, j) in [
            ("column convolutions", mono.column_convolutions, joint.column_convolutions),
            ("frame responses", mono.frame_responses, joint.frame_responses),
            ("frame transforms", mono.frame_transforms, joint.frame_transforms),
        ] {
            if m == 0 && j == 0 {
                continue;
            }
            if j != 2 * m {
                return Err(format!(
                    "{what}: mono {m}, joint {j} — expected {}, and \
                     {} would be a third analysis",
                    2 * m,
                    3 * m
                ));
            }
        }
        Ok(())
    }

    /// Two channels' work for a stereo run, not three — counted where the work
    /// is dispatched, and in each unit separately.
    #[test]
    fn a_joint_run_does_two_channel_passes_and_not_three() {
        let sr = 16_000u32;
        let n = (1.0 * sr as f32) as usize;
        let tau = std::f32::consts::TAU;
        let l: Vec<f32> = (0..n).map(|i| 0.5 * (tau * 300.0 * i as f32 / sr as f32).sin()).collect();
        let r: Vec<f32> = (0..n).map(|i| 0.4 * (tau * 900.0 * i as f32 / sr as f32).sin()).collect();
        let hop = hop_for_fps(sr, 60.0);
        let cfg = cfg(1.0, 1.2, 4, 0.5);
        let (lo, hi, bars) = (40.0f32, 5_000.0, 64);

        // Read off the result, not off a process-wide counter: the suite runs
        // analyses in parallel, and a static reported whichever run finished
        // last — which is how a joint run once measured as doing a *third* of a
        // mono run's work.
        let one = analyze_mono_work(&l, sr, bars, lo, hi, hop, &cfg);
        let joint = analyze_joint_via(never_gpu, &l, &r, sr, bars, lo, hi, hop, &cfg);
        let two = joint.work;
        assert!(!joint.mix.is_empty());
        assert!(
            one.column_convolutions > 0 && one.frame_responses > 0,
            "setup: this configuration should exercise both the transform and \
             the direct routes, got {one:?}"
        );

        println!("  mono  {one:?}");
        println!("  joint {two:?}");
        two_channel_contract(one, two).unwrap_or_else(|e| panic!("{e}"));

        // And the check has teeth: the same run with one deliberate extra
        // analysis of the mean signal is rejected.
        let mutant = analyze_joint_seamed(
            never_gpu,
            Seams { redundant_mix_transform: true, ..Seams::production() },
            &l, &r, sr, bars, lo, hi, hop, &cfg,
        );
        println!("  mutant {:?}", mutant.work);
        let refused = two_channel_contract(one, mutant.work)
            .expect_err("a third analysis was accepted by the contract check");
        println!("  contract refuses it: {refused}");
        assert_eq!(
            mutant.work.column_convolutions, 3 * one.column_convolutions,
            "the mutation did not actually add a third column convolution"
        );
        assert_eq!(
            mutant.work.frame_responses, 3 * one.frame_responses,
            "the mutation did not actually add a third frame response"
        );

        // The counts are not the only evidence, and deliberately so: an
        // increment beside a no-op looks exactly like an increment beside a
        // convolution. The extra work is a real analysis of a real mean signal,
        // its result is consumed, and it agrees with the derived mix — which is
        // both proof that it ran and an independent re-check of the derivation.
        let dev = mutant
            .redundant_mix_deviation
            .expect("the redundant analysis produced no result to check");
        let scale = peak_of(&mutant.mix).max(1e-12);
        println!("  redundant mean analysis vs derived mix: {dev:.3e} of {scale:.3}");
        assert!(
            dev / scale < 2e-3,
            "the redundant mean-signal analysis disagreed with the derived mix              by {dev:.3e} of {scale:.3}"
        );

        // And the mutation changes only the work, never the answer.
        assert_eq!(mutant.mix.len(), joint.mix.len());
        assert!(worst_rel(&mutant.mix, &joint.mix) < 1e-6, "the mutation changed the mix");
        assert!(worst_rel(&mutant.left, &joint.left) < 1e-6);
        assert!(worst_rel(&mutant.right, &joint.right) < 1e-6);

        // A run without the seam records nothing, so the field cannot be a
        // leftover from some other run.
        assert!(joint.redundant_mix_deviation.is_none());
        assert!(analyze_mono_work(&l, sr, bars, lo, hi, hop, &cfg).column_convolutions > 0);
    }

    /// A block size the budget cannot hold falls back to the per-kernel
    /// transform, for both channels, and the answer does not change.
    ///
    /// This is a driver-level path that no earlier test reached: the shared
    /// block set is normally built for every size a group needs, and the
    /// fallback only happens when memory runs out. A budget of zero forces it
    /// through the real orchestration rather than through a reimplementation of
    /// the decision.
    #[test]
    fn a_budget_too_small_for_shared_blocks_falls_back_for_both_channels() {
        let sr = 16_000u32;
        let n = (2.0 * sr as f32) as usize;
        let tau = std::f32::consts::TAU;
        let l: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.5 * (tau * 55.0 * t).sin() + 0.3 * (tau * 900.0 * t).sin()
            })
            .collect();
        let r: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.45 * (tau * 55.0 * t + 1.3).sin() + 0.2 * (tau * 2_500.0 * t).sin()
            })
            .collect();
        let hop = hop_for_fps(sr, 60.0);
        let (lo, hi, bars) = (30.0f32, 6_000.0, 96);
        let cfg = cfg(1.0, 1.2, 4, 0.5);

        let starved = analyze_joint_seamed(
            never_gpu,
            Seams { shared_budget: 0, ..Seams::production() },
            &l, &r, sr, bars, lo, hi, hop, &cfg,
        );
        let normal = analyze_joint_via(never_gpu, &l, &r, sr, bars, lo, hi, hop, &cfg);
        assert!(!starved.mix.is_empty());
        assert_eq!(starved.mix.len(), normal.mix.len());

        // The fallback is a different route, not a different answer — and all
        // three outputs are held to independent analyses, not merely to each
        // other.
        let om = analyze_via(never_gpu, &mean_of(&l, &r), sr, bars, lo, hi, hop, &cfg);
        let ol = analyze_via(never_gpu, &l, sr, bars, lo, hi, hop, &cfg);
        let or = analyze_via(never_gpu, &r, sr, bars, lo, hi, hop, &cfg);
        let fs = peak_of(&om).max(peak_of(&ol)).max(peak_of(&or));
        for (what, got, want) in [
            ("mix", &starved.mix, &om),
            ("left", &starved.left, &ol),
            ("right", &starved.right, &or),
        ] {
            let worst = worst_rel_to(got, want, fs);
            println!("  starved budget, {what:<6} vs oracle: worst {worst:.2e}");
            assert!(worst < 2e-3, "{what} differs by {worst:.3e} with no shared blocks");
        }

        // The contract still holds on the fallback route.
        let mono = analyze_mono_work(&l, sr, bars, lo, hi, hop, &cfg);
        let starved_mono = analyze_routed(
            Input::Mono(&l), sr, bars, lo, hi, hop, &cfg, &|| true, &|_| {}, never_gpu,
            Seams { shared_budget: 0, ..Seams::production() },
        )
        .work;
        println!("  shared {mono:?}");
        println!("  starved mono {starved_mono:?}, joint {:?}", starved.work);
        two_channel_contract(starved_mono, starved.work).unwrap_or_else(|e| panic!("{e}"));
    }

    /// A device that declines mid-run — including left succeeding and right
    /// failing, with an earlier chunk already staged.
    ///
    /// The failure is injected at the production dispatch boundary, so what is
    /// under test is the real orchestration: whether the already-staged chunk
    /// is still drained, whether the bars the failed chunk should have produced
    /// are recovered by the sweep after the join, and whether the result is
    /// right.
    /// The group loop's routing decisions belong to their groups.
    ///
    /// This is the call site, not the mechanism: `gpu_calib`'s own suite covers
    /// what a scope does, and nothing covered the loop that opens one. A mutant
    /// that replaces `drop(group_scope)` with `mem::forget` survived the whole
    /// calibration suite, because every one of those tests opens its own scope.
    ///
    /// An analysis that has returned must leave nothing parked on the thread.
    /// Anything left there is a decision the next group would consume, and its
    /// time would be filed under this analysis's block size.
    #[test]
    fn an_analysis_leaves_no_routing_decision_parked() {
        // The fixture has to produce a group at or above the routing floor
        // (2^17), or `use_device` declines every size, nothing is ever parked,
        // and the test would pass by vacuum. `fft_block_len(k)` is
        // `(2k).next_power_of_two()`, so a 2^17 group needs a kernel of
        // 32 769..=65 536 taps — which means a window of roughly 0.7 s at
        // 48 kHz, and a signal comfortably longer than that.
        // 256 bars is the smallest count whose grid q produces a 2^17 group at
        // these bounds; 128 tops out at 2^16. The block size follows from the
        // grid's own q, which scales with the bar count — the window cap only
        // ever shortens a kernel, so widening the window does not help.
        let (sr, bars, lo, hi) = (48_000u32, 256usize, 40.0f32, 18_000.0f32);
        let n = (sr as f32 * 4.0) as usize;
        let sig: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.0023).sin() * 0.7 + (i as f32 * 0.019).sin() * 0.3)
            .collect();
        let cfg = AsltConfig { max_window_s: 0.9, ..AsltPreset::Fast.config() };
        let hop = hop_for_fps(sr, 120.0);

        // The route must go through the calibration, or nothing is ever
        // parked and this test cannot fail. `route_to_device` short-circuits to
        // `true` in a test build without consulting `use_device`, so a test
        // that used it would pass under any mutation of the scope discipline —
        // which is exactly what the first version of this test did.
        fn route_via_calibration(n: usize) -> bool {
            super::super::gpu_calib::use_device(n)
        }

        let _env = super::super::gpu_calib::aslt_test_env("scope_discipline");

        // The check has to happen **on the pool thread**. `install` runs its
        // closure on a rayon worker and the parked decision is thread-local, so
        // asking from the test thread always sees an empty slot and the
        // assertion would hold under any mutation of the scope discipline.
        let (produced, clean) = super::super::gpu_calib::install(|| {
            let out = analyze_via(route_via_calibration, &sig, sr, bars, lo, hi, hop, &cfg);
            let clean = super::super::gpu_calib::pending_is_empty();
            (out, clean)
        });
        assert!(!produced.is_empty(), "setup: the analysis produced nothing");
        assert!(
            _env.saw_a_decision(),
            "setup: no group ever took a routing decision, so nothing could leak"
        );
        assert!(clean, "a completed analysis left a routing decision parked on the thread");

        // And one cancelled part-way, which is the case that skips `record`
        // entirely and is the one that actually leaks: the group loop breaks at
        // the top of the *next* iteration, so no later `begin_group` ever runs
        // to clear what the abandoned group left.
        let seen = std::sync::atomic::AtomicUsize::new(0);
        let clean = super::super::gpu_calib::install(|| {
            let _ = analyze_routed(
                Input::Mono(&sig), sr, bars, lo, hi, hop, &cfg,
                &|| seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 2,
                &|_| {}, route_via_calibration, Seams::production(),
            );
            super::super::gpu_calib::pending_is_empty()
        });
        assert!(clean, "a cancelled analysis left a routing decision parked on the thread");
    }

    #[test]
    fn a_device_that_declines_midway_still_produces_the_whole_analysis() {
        if super::super::gpu::GpuFft::shared().is_none() {
            println!("no GPU adapter — device failure injection not exercised here");
            return;
        }
        let sr = 16_000u32;
        let n = (4.0 * sr as f32) as usize;
        let tau = std::f32::consts::TAU;
        let l: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.5 * (tau * 45.0 * t).sin() + 0.2 * (tau * 900.0 * t).sin()
            })
            .collect();
        let r: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.4 * (tau * 45.0 * t + 0.7).sin() + 0.3 * (tau * 300.0 * t).sin()
            })
            .collect();
        let hop = hop_for_fps(sr, 60.0);
        let (lo, hi, bars) = (20.0f32, 8_000.0, 128);
        let cfg = cfg(1.0, 1.2, 4, 0.5);

        // A tap budget small enough to split a group into several chunks. At
        // the production budget this configuration is one chunk, and a failure
        // on the first chunk never meets the case worth testing: a chunk
        // already staged, waiting to be assembled, when a later dispatch fails.
        let chunked = Seams { kernel_tap_budget: 1 << 20, ..Seams::production() };

        let clean = analyze_joint_seamed(
            eligible_gpu, chunked, &l, &r, sr, bars, lo, hi, hop, &cfg,
        );
        assert!(!clean.mix.is_empty(), "setup: the device run produced nothing");

        // Left succeeds, right fails, and not on the first chunk.
        let seen = std::sync::Mutex::new(Vec::<(usize, usize)>::new());
        let fault = |chunk: usize, channel: usize| -> bool {
            seen.lock().unwrap_or_else(|e| e.into_inner()).push((chunk, channel));
            chunk >= 1 && channel == 1
        };
        let hurt = analyze_joint_seamed(
            eligible_gpu,
            Seams { dispatch_fault: Some(&fault), ..chunked },
            &l, &r, sr, bars, lo, hi, hop, &cfg,
        );

        let dispatches = seen.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let failures = hurt.work.device_failures;
        let chunks = dispatches.iter().map(|(c, _)| *c).max().map(|m| m + 1).unwrap_or(0);
        println!("  dispatches {} over {chunks} chunk(s); device failures {failures}",
                 dispatches.len());
        assert!(
            chunks >= 2,
            "setup: the budget did not split the group; only {chunks} chunk(s) ran, \
             so no chunk was ever staged when a failure landed"
        );
        assert!(failures > 0, "setup: the injected failure never fired");
        // Left really did succeed before right failed, on the failing chunk.
        assert!(
            dispatches.contains(&(1, 0)) && dispatches.contains(&(1, 1)),
            "expected both channels dispatched on chunk 1, saw {dispatches:?}"
        );

        // Whatever the device did, the analysis is complete and correct.
        assert_eq!(hurt.mix.len(), clean.mix.len(), "frames lost to the failure");
        assert_eq!(hurt.left.len(), clean.left.len());
        assert_eq!(hurt.right.len(), clean.right.len());
        assert_eq!(hurt.mix[0].len(), bars, "bars lost to the failure");
        for (what, a, b) in [
            ("mix", &hurt.mix, &clean.mix),
            ("left", &hurt.left, &clean.left),
            ("right", &hurt.right, &clean.right),
        ] {
            let worst = worst_rel(a, b);
            println!("  after failure, {what:<6} vs the clean run: worst {worst:.2e}");
            assert!(worst < 1e-3, "{what} differs by {worst:.3e} after a device failure");
        }

        // A failed dispatch costs a retry on the cores. That is real physical
        // work, so the two-channel contract does not hold on raw counts — and
        // the check says so rather than pretending otherwise.
        let mono = analyze_mono_work(&l, sr, bars, lo, hi, hop, &cfg);
        println!("  clean {:?}", clean.work);
        println!("  hurt  {:?}", hurt.work);
        assert!(
            two_channel_contract(mono, hurt.work).is_err(),
            "a run with retries must not claim an exact 2x physical total"
        );
        assert!(
            hurt.work.column_convolutions > clean.work.column_convolutions,
            "the retry after a failed dispatch should show as extra column \
             convolutions: clean {}, hurt {}",
            clean.work.column_convolutions,
            hurt.work.column_convolutions,
        );
    }

    /// The same, on the device, when there is one.
    #[test]
    fn the_derived_mix_matches_on_the_device_too() {
        if super::super::gpu::GpuFft::shared().is_none() {
            println!("no GPU adapter — device route not exercised on this machine");
            return;
        }
        let sr = 16_000u32;
        let n = (4.0 * sr as f32) as usize;
        let tau = std::f32::consts::TAU;
        let l: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.5 * (tau * 45.0 * t).sin() + 0.2 * (tau * 900.0 * t).sin()
            })
            .collect();
        let r: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.4 * (tau * 45.0 * t + 0.7).sin() + 0.3 * (tau * 300.0 * t).sin()
            })
            .collect();
        let hop = hop_for_fps(sr, 60.0);
        let (lo, hi, bars) = (20.0f32, 8_000.0, 128);
        let cfg = cfg(1.0, 1.2, 4, 0.5);

        let dev = analyze_joint_via(eligible_gpu, &l, &r, sr, bars, lo, hi, hop, &cfg);
        let cpu = analyze_joint_via(never_gpu, &l, &r, sr, bars, lo, hi, hop, &cfg);
        assert!(!dev.mix.is_empty());

        for (what, a, b) in [
            ("mix", &dev.mix, &cpu.mix),
            ("left", &dev.left, &cpu.left),
            ("right", &dev.right, &cpu.right),
        ] {
            let worst = worst_rel(a, b);
            println!("  device vs cores, {what:<6} worst relative {worst:.2e}");
            assert!(worst < 1e-3, "{what}: device and cores differ by {worst:.3e}");
        }

        // And the device's mix is still the mix, not an average of magnitudes.
        let oracle = analyze_via(eligible_gpu, &mean_of(&l, &r), sr, bars, lo, hi, hop, &cfg);
        let worst = worst_rel(&dev.mix, &oracle);
        println!("  device mix vs oracle: worst relative {worst:.2e}");
        assert!(worst < 2e-3, "the device's derived mix differs by {worst:.3e}");
    }

    /// Edge frames — where the envelope is renormalised over the taps that
    /// survive — combine correctly too. They are the frames where the two
    /// routes differ most, and where a divisor applied on the wrong side of the
    /// average would show.
    #[test]
    fn the_derived_mix_matches_at_the_signal_edges() {
        let sr = 8_000u32;
        let n = (0.5 * sr as f32) as usize;
        let tau = std::f32::consts::TAU;
        let l: Vec<f32> = (0..n).map(|i| 0.6 * (tau * 120.0 * i as f32 / sr as f32).sin()).collect();
        let r: Vec<f32> = (0..n).map(|i| 0.5 * (tau * 120.0 * i as f32 / sr as f32 + 2.0).sin()).collect();
        let hop = hop_for_fps(sr, 120.0);
        let cfg = cfg(1.0, 1.2, 4, 0.5);
        let (lo, hi, bars) = (40.0f32, 2_000.0, 48);

        let joint = analyze_joint_via(never_gpu, &l, &r, sr, bars, lo, hi, hop, &cfg);
        let oracle = analyze_via(never_gpu, &mean_of(&l, &r), sr, bars, lo, hi, hop, &cfg);
        let last = joint.mix.len() - 1;
        for fi in [0usize, 1, last - 1, last] {
            let peak = oracle[fi].iter().cloned().fold(0.0f32, f32::max).max(1e-12);
            let worst = joint.mix[fi].iter().zip(&oracle[fi])
                .map(|(a, b)| (a - b).abs() / peak)
                .fold(0.0f32, f32::max);
            assert!(worst < 2e-3, "edge frame {fi} differs by {worst:.3e}");
        }
    }

    /// The complex leaves agree with each other, which is what lets the
    /// magnitude ones be defined as their norms.
    #[test]
    fn the_complex_routes_agree_with_each_other() {
        let sr = 8_000f32;
        let n = 200_000usize;
        let tau = std::f32::consts::TAU;
        let sig: Vec<f32> = (0..n)
            .map(|i| 0.7 * (tau * 60.0 * i as f32 / sr).sin())
            .collect();
        let hop = 64usize;
        let frames = frame_count(sig.len(), hop);
        // Long enough to take the transform route, which is the point: the
        // three routes have to agree, and two of them only exist for kernels
        // like this one.
        let m = Morlet::new(60.0, 40.0, sr);
        assert!(
            fft_is_cheaper(m.taps(), hop),
            "setup: a {}-tap kernel should prefer a transform at hop {hop}",
            m.taps(),
        );
        assert!(n > m.taps(), "setup: the signal must be longer than the kernel");

        let direct: Vec<C32> =
            (0..frames).map(|fi| m.response_c(&sig, (fi * hop) as isize)).collect();
        let own = m.complex_via_fft(&sig, hop, frames).expect("own-transform route declined");
        let blocks = SignalBlocks::build(&sig, fft_block_len(m.taps()));
        let shared = m
            .complex_via_shared(&sig, &blocks, hop, frames)
            .expect("shared route declined");

        let peak = direct.iter().map(|c| c.norm()).fold(0.0f32, f32::max).max(1e-12);
        let worst = |a: &[C32], b: &[C32]| -> f32 {
            a.iter().zip(b)
                .map(|(x, y)| ((x.re - y.re).hypot(x.im - y.im)) / peak)
                .fold(0.0f32, f32::max)
        };
        let (w1, w2) = (worst(&own, &direct), worst(&shared, &direct));
        println!("  complex routes vs direct: own {w1:.2e}, shared {w2:.2e}");
        assert!(w1 < 1e-4, "own-transform complex differs by {w1:.3e}");
        assert!(w2 < 1e-4, "shared-transform complex differs by {w2:.3e}");

        // And the magnitude route is exactly `norms` of the complex one — the
        // same values, bit for bit, because there is only one convolution and
        // the magnitude is defined as its norm.
        //
        // `sqrt(re² + im²)` and not `Complex::norm`, which is `hypot`: hypot is
        // the more accurate of the two and would have been the better choice in
        // a new transform, but this one has produced `sqrt(re² + im²)` since
        // 1.4 and every cached analysis on the owner's disk was written with
        // it. Changing it here would shift the mono spectrum for no reason
        // anyone asked for.
        let mags = m.magnitudes_via_shared(&sig, &blocks, hop, frames).unwrap();
        for (mg, c) in mags.iter().zip(&shared) {
            assert_eq!(
                *mg,
                (c.re * c.re + c.im * c.im).sqrt(),
                "the magnitude route is not the norm of the complex one"
            );
        }
    }

    // ── Reused transform scratch ────────────────────────────────────────────
    //
    // The three block loops hand one scratch buffer to every transform instead
    // of letting `process` allocate a fresh one per call. The `reference_*`
    // functions are those loops as they were before, `process` and all, so
    // these tests compare against the original form rather than against
    // themselves. Equality is `to_bits` on both components: a float `==` would
    // accept -0.0 for 0.0 and reject a NaN that both forms produced.

    /// A kernel of exactly `k` taps with deterministic, uneven values, so a
    /// test can pick its block size directly. Not a wavelet: these tests are
    /// about bits, not about spectra.
    fn synthetic_morlet(k: usize, seed: u64) -> Morlet {
        assert!(k % 2 == 1, "setup: a kernel table has an odd number of taps");
        let mut next = noise_source(seed);
        let re = (0..k).map(|_| next()).collect();
        let im = (0..k).map(|_| next()).collect();
        let raw: Vec<f32> = (0..k).map(|_| next().abs() + 0.01).collect();
        let total: f32 = raw.iter().sum();
        Morlet { re, im, env: raw.iter().map(|v| v / total).collect(), half: k / 2 }
    }

    fn noise_source(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / 8_388_608.0) - 1.0
        }
    }

    fn noise(len: usize, seed: u64) -> Vec<f32> {
        let mut next = noise_source(seed);
        (0..len).map(|_| next()).collect()
    }

    /// The fused mix magnitude against the two calls it replaced, bit for bit.
    ///
    /// Covers the shapes the fold can meet: ordinary signal, channels equal,
    /// channels in exact antiphase (where the mix is zero and the sign of the
    /// zero matters), one channel silent, and values small enough that the
    /// squares underflow.
    #[test]
    fn the_fused_mix_magnitude_matches_the_two_step_form_bit_for_bit() {
        let mut next = noise_source(0xA5A5);
        let cases: Vec<(&str, Vec<C32>, Vec<C32>)> = {
            let base: Vec<C32> = (0..4096)
                .map(|_| C32 { re: next(), im: next() })
                .collect();
            let tiny: Vec<C32> = base.iter()
                .map(|c| C32 { re: c.re * 1e-30, im: c.im * 1e-30 })
                .collect();
            let zero = vec![C32 { re: 0.0, im: 0.0 }; base.len()];
            let anti: Vec<C32> = base.iter()
                .map(|c| C32 { re: -c.re, im: -c.im })
                .collect();
            let other: Vec<C32> = (0..base.len())
                .map(|_| C32 { re: next(), im: next() })
                .collect();
            vec![
                ("independent", base.clone(), other),
                ("identical", base.clone(), base.clone()),
                ("exact antiphase", base.clone(), anti),
                ("one channel silent", base.clone(), zero),
                ("underflowing squares", tiny.clone(), tiny),
            ]
        };
        for (what, l, r) in cases {
            let got = mix_norms(&l, &r);
            let want = norms(mix_complex(&l, &r));
            assert_eq!(got.len(), want.len(), "{what}: lengths differ");
            if let Some(i) = got.iter().zip(&want).position(|(a, b)| a.to_bits() != b.to_bits()) {
                panic!("{what}: first difference at {i}: {} against {}", got[i], want[i]);
            }
        }
    }

    fn assert_same_bits(what: &str, got: &[C32], want: &[C32]) {
        assert_eq!(got.len(), want.len(), "{what}: lengths differ");
        if let Some(i) = got.iter().zip(want).position(|(a, b)| {
            a.re.to_bits() != b.re.to_bits() || a.im.to_bits() != b.im.to_bits()
        }) {
            panic!("{what}: first difference at {i} of {}: {:?} vs {:?}", got.len(), got[i], want[i]);
        }
    }

    /// [`SignalBlocks::build`] before the scratch was reused.
    fn reference_signal_blocks(signal: &[f32], n: usize) -> SignalBlocks {
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
        SignalBlocks { n, stride, data }
    }

    /// [`Morlet::complex_via_shared`] before the scratch was reused.
    fn reference_complex_via_shared(
        m: &Morlet, signal: &[f32], blocks: &SignalBlocks, hop: usize, frames: usize,
    ) -> Option<Vec<C32>> {
        use rustfft::num_complex::Complex;

        let k = m.re.len();
        let n_sig = signal.len();
        let half = m.half as isize;
        let n = blocks.n;
        if n_sig <= k || hop == 0 || frames == 0 || n <= k { return None; }
        if blocks.stride > n - k + 1 { return None; }

        let first = (half as usize).div_ceil(hop);
        let last_pos = (n_sig as isize) - 1 - half;
        if last_pos < 0 { return None; }
        let last = ((last_pos as usize) / hop).min(frames.saturating_sub(1));
        if first > last { return None; }

        let inv = ASLT_PLANNER.with(|p| p.borrow_mut().plan_fft_inverse(n));

        let mut kernel = vec![Complex { re: 0.0f32, im: 0.0f32 }; n];
        for (slot, i) in kernel.iter_mut().zip((0..k).rev()) {
            *slot = Complex { re: m.re[i], im: m.im[i] };
        }
        ASLT_PLANNER.with(|p| p.borrow_mut().plan_fft_forward(n)).process(&mut kernel);

        let mut out = vec![C32 { re: 0.0, im: 0.0 }; frames];
        for (fi, slot) in out.iter_mut().enumerate() {
            if fi < first || fi > last {
                *slot = m.response_c(signal, (fi * hop) as isize);
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
                    *slot = C32 { re: 2.0 * scale * c.re, im: 2.0 * scale * c.im };
                }
            }
        }
        Some(out)
    }

    /// [`Morlet::complex_via_fft`] before the scratch was reused.
    fn reference_complex_via_fft(
        m: &Morlet, signal: &[f32], hop: usize, frames: usize,
    ) -> Option<Vec<C32>> {
        use rustfft::num_complex::Complex;

        let k = m.re.len();
        let n_sig = signal.len();
        let half = m.half as isize;
        if n_sig <= k || hop == 0 || frames == 0 { return None; }

        let n = fft_block_len(k);
        if n <= k { return None; }
        let step = n - k + 1;

        let first = (half as usize).div_ceil(hop);
        let last_pos = (n_sig as isize) - 1 - half;
        if last_pos < 0 { return None; }
        let last = ((last_pos as usize) / hop).min(frames.saturating_sub(1));
        if first > last { return None; }

        let (fwd, inv) = ASLT_PLANNER.with(|p| {
            let mut p = p.borrow_mut();
            (p.plan_fft_forward(n), p.plan_fft_inverse(n))
        });

        let mut kernel = vec![Complex { re: 0.0f32, im: 0.0f32 }; n];
        for (slot, i) in kernel.iter_mut().zip((0..k).rev()) {
            *slot = Complex { re: m.re[i], im: m.im[i] };
        }
        fwd.process(&mut kernel);

        let mut out = vec![C32 { re: 0.0, im: 0.0 }; frames];
        for (fi, slot) in out.iter_mut().enumerate() {
            if fi < first || fi > last {
                *slot = m.response_c(signal, (fi * hop) as isize);
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
                    *slot = C32 { re: 2.0 * scale * c.re, im: 2.0 * scale * c.im };
                }
            }
            base += step;
        }
        Some(out)
    }

    /// The direct loops against the form they replaced, bit for bit.
    ///
    /// `response_parts` used to index the signal once per tap; it now takes the
    /// window as a slice and zips it. The claim is that nothing but a bounds
    /// check went, so the two forms are held to identical `f64` accumulators
    /// over every shape the window can take against a signal — wholly inside
    /// it, hanging off either end, covering the whole of it at once, and
    /// missing it entirely. The shapes are *counted*, so a centre list that
    /// quietly stopped reaching one of them fails the test rather than
    /// narrowing it.
    #[test]
    fn the_direct_window_matches_the_indexed_form_bit_for_bit() {
        /// `Morlet::response_parts` before the window was taken as a slice.
        fn reference(m: &Morlet, signal: &[f32], centre: isize) -> (f64, f64, f64) {
            let n = signal.len() as isize;
            let base = centre - m.half as isize;
            let end = base + m.re.len() as isize;
            let lo = base.max(0);
            let hi = end.min(n);
            if hi <= lo { return (0.0, 0.0, 0.0); }

            let mut acc_re = 0.0f64;
            let mut acc_im = 0.0f64;

            if base >= 0 && end <= n {
                for (k, (&wr, &wi)) in m.re.iter().zip(m.im.iter()).enumerate() {
                    let s = signal[(base + k as isize) as usize];
                    acc_re += (s * wr) as f64;
                    acc_im += (s * wi) as f64;
                }
                return (acc_re, acc_im, 1.0);
            }

            let mut env_sum = 0.0f64;
            for i in lo..hi {
                let k = (i - base) as usize;
                let s = signal[i as usize];
                acc_re += (s * m.re[k]) as f64;
                acc_im += (s * m.im[k]) as f64;
                env_sum += m.env[k] as f64;
            }
            (acc_re, acc_im, env_sum)
        }

        let (mut interior, mut left, mut right, mut spanning, mut missed) = (0, 0, 0, 0, 0);

        for (k, seed) in [(65usize, 1u64), (1025, 2), (4097, 3)] {
            let m = synthetic_morlet(k, seed);
            let half = m.half as isize;
            // One signal longer than the kernel and one shorter, so the window
            // can also hang off both ends of the same signal.
            for len in [k * 3 + 7, k / 2] {
                let sig = noise(len, seed * 31 + len as u64);
                let n = len as isize;
                let mut centres = vec![-half - 1, -half, -half + 1, -1, 0, 1, half - 1, half, half + 1];
                centres.extend([n - half - 1, n - half, n - half + 1, n - 1, n, n + half]);
                centres.extend([n / 3, n / 2, n / 2 + 1]);

                for c in centres {
                    let got = m.response_parts(&sig, c);
                    let want = reference(&m, &sig, c);
                    let bits = |t: (f64, f64, f64)| (t.0.to_bits(), t.1.to_bits(), t.2.to_bits());
                    assert_eq!(bits(got), bits(want),
                               "k={k} len={len} centre={c}: {got:?} against {want:?}");

                    // And through the reader that scales and divides them, so
                    // the magnitude a bar actually stores is covered too.
                    let (re, im, div) = want;
                    let want_mag = if div <= 0.0 {
                        0.0f32
                    } else {
                        (2.0 * (re * re + im * im).sqrt() / div) as f32
                    };
                    assert_eq!(m.response(&sig, c).to_bits(), want_mag.to_bits(),
                               "k={k} len={len} centre={c}: magnitude");

                    let base = c - half;
                    let end = base + k as isize;
                    if end <= 0 || base >= n { missed += 1 }
                    else if base >= 0 && end <= n { interior += 1 }
                    else if base < 0 && end > n { spanning += 1 }
                    else if base < 0 { left += 1 }
                    else { right += 1 }
                }
            }
        }

        for (what, seen) in [("interior", interior), ("left edge", left),
                             ("right edge", right), ("spanning both ends", spanning),
                             ("missing the signal", missed)] {
            assert!(seen > 0, "setup: no {what} case was exercised");
        }
    }

    /// The premise the loops rely on: a scratch buffer holding someone else's
    /// numbers — here NaN and huge values, then whatever a previous transform
    /// left — gives the same bits as the fresh zeroed one `process` allocates.
    #[test]
    fn transform_scratch_contents_never_reach_the_result() {
        use rustfft::num_complex::Complex;
        for log2 in [13u32, 15, 17, 19, FFT_MAX_LOG2] {
            let n = 1usize << log2;
            let input: Vec<Complex<f32>> = noise(2 * n, u64::from(log2))
                .chunks_exact(2)
                .map(|c| Complex { re: c[0], im: c[1] })
                .collect();
            let (fwd, inv) = ASLT_PLANNER.with(|p| {
                let mut p = p.borrow_mut();
                (p.plan_fft_forward(n), p.plan_fft_inverse(n))
            });
            for (dir, plan) in [("forward", &fwd), ("inverse", &inv)] {
                let len = plan.get_inplace_scratch_len();
                assert!(len > 0, "2^{log2} {dir}: no scratch needed, so this proves nothing");
                let mut want = input.clone();
                plan.process(&mut want);

                let garbage = [f32::NAN, 1.0e30, -7.5, f32::INFINITY];
                let mut scratch: Vec<Complex<f32>> = (0..len)
                    .map(|i| Complex { re: garbage[i % 4], im: garbage[(i + 1) % 4] })
                    .collect();
                for pass in ["planted garbage", "a previous transform's leftovers"] {
                    let mut got = input.clone();
                    plan.process_with_scratch(&mut got, &mut scratch);
                    assert_same_bits(&format!("2^{log2} {dir}, scratch holding {pass}"), &got, &want);
                }
            }
        }
    }

    /// Each loop against its original form, bit for bit.
    ///
    /// The three `reference_*` helpers are deliberately *stale*: they hold the
    /// shape each loop had before it was optimised, which is the only thing
    /// that makes this a check rather than a tautology. They currently cover
    /// three changes — the reused transform scratch, the block fill written as
    /// a bounded copy instead of a checked lookup per element
    /// (`SignalBlocks::build` and `complex_via_fft`), and the shared route
    /// multiplying the block in as it reads it instead of copying it first.
    /// Do not update them to match the production loops.
    ///
    /// Coverage is a selection, not every size the planner can reach: five
    /// transform lengths (2^13, 2^15, 2^17, 2^19, 2^21 — the smallest a kernel
    /// plans and `FFT_MAX_LOG2`, with three in between), each with a signal
    /// shorter than one block, exactly one block and an odd tail at 2^13–2^17,
    /// and the odd tail alone at 2^19 and 2^21 where a debug transform is slow.
    /// A second channel and the joint mix are covered at 2^13–2^17.
    #[test]
    fn reused_scratch_loops_match_the_process_form_bit_for_bit() {
        // Short, complete and odd-tail signals at the smaller sizes, both
        // channels; the odd tail alone at the two largest, where a debug
        // transform is slow.
        let full = [13u32, 15, 17];
        for log2 in [13u32, 15, 17, 19, FFT_MAX_LOG2] {
            let n = 1usize << log2;
            let k = n / 2 - 1;
            let m = synthetic_morlet(k, u64::from(log2));
            assert_eq!(fft_block_len(k), n, "setup: {k} taps must plan a 2^{log2} transform");
            let hop = n / 8;
            let odd_tail = n + n / 3 + 1;
            let lengths: &[(&str, usize)] = if full.contains(&log2) {
                &[("shorter than a block", k + k / 2), ("one block", n), ("odd tail", odd_tail)]
            } else {
                &[("odd tail", odd_tail)]
            };
            let channels: &[u64] = if full.contains(&log2) { &[1, 2] } else { &[1] };
            for &(shape, len) in lengths {
                let frames = frame_count(len, hop);
                let mut cols = Vec::new();
                for &ch in channels {
                    let what = |route: &str| format!("2^{log2}, {shape} ({len}), channel {ch}, {route}");
                    let sig = noise(len, u64::from(log2) * 10 + ch);

                    let blocks = SignalBlocks::build(&sig, n);
                    let ref_blocks = reference_signal_blocks(&sig, n);
                    assert_eq!((blocks.n, blocks.stride), (ref_blocks.n, ref_blocks.stride));
                    assert_same_bits(&what("signal blocks"), &blocks.data, &ref_blocks.data);

                    let shared = m.complex_via_shared(&sig, &blocks, hop, frames)
                        .unwrap_or_else(|| panic!("{}: declined", what("shared route")));
                    let want_shared = reference_complex_via_shared(&m, &sig, &ref_blocks, hop, frames)
                        .expect("reference shared route declined");
                    assert_same_bits(&what("shared route"), &shared, &want_shared);

                    let own = m.complex_via_fft(&sig, hop, frames)
                        .unwrap_or_else(|| panic!("{}: declined", what("own-transform route")));
                    let want_own = reference_complex_via_fft(&m, &sig, hop, frames)
                        .expect("reference own-transform route declined");
                    assert_same_bits(&what("own-transform route"), &own, &want_own);

                    cols.push([shared, want_shared, own, want_own]);
                }
                // The joint producer mixes the two channels' complex columns
                // before anything nonlinear, so the mix has to match too.
                if let [l, r] = &cols[..] {
                    assert_same_bits(&format!("2^{log2}, {shape}, shared mix"),
                                     &mix_complex(&l[0], &r[0]), &mix_complex(&l[1], &r[1]));
                    assert_same_bits(&format!("2^{log2}, {shape}, own-transform mix"),
                                     &mix_complex(&l[2], &r[2]), &mix_complex(&l[3], &r[3]));
                }
            }
        }
    }

    /// Allocations made on this thread while `f` runs.
    fn allocations_in(f: impl FnOnce()) -> u64 {
        let before = crate::alloc_probe::arm();
        f();
        crate::alloc_probe::disarm(before)
    }

    /// The fixture for the allocation tests: one transform size, and two
    /// signals four times apart in length, so the loop runs four times as many
    /// transforms on the second. Plans for the size are built on this thread
    /// before anything is counted.
    struct ScratchFixture {
        m: Morlet,
        n: usize,
        hop: usize,
        short: Vec<f32>,
        long: Vec<f32>,
    }

    impl ScratchFixture {
        fn new() -> Self {
            let n = 1usize << 13;
            let (fwd, inv) = ASLT_PLANNER.with(|p| {
                let mut p = p.borrow_mut();
                (p.plan_fft_forward(n), p.plan_fft_inverse(n))
            });
            assert!(
                fwd.get_inplace_scratch_len() > 0 && inv.get_inplace_scratch_len() > 0,
                "setup: a transform that needs no scratch allocates none either way",
            );
            Self {
                m: synthetic_morlet(n / 2 - 1, 7),
                n,
                hop: n / 8,
                short: noise(3 * n + 1, 8),
                long: noise(12 * n + 1, 8),
            }
        }
    }

    /// `SignalBlocks::build` makes the same allocations whatever the length.
    #[test]
    fn building_signal_blocks_does_not_allocate_per_transform() {
        let fx = ScratchFixture::new();
        // Warm: one untimed call of each form on this thread.
        std::hint::black_box(SignalBlocks::build(&fx.short, fx.n));
        std::hint::black_box(reference_signal_blocks(&fx.short, fx.n));

        let count = |sig: &[f32], reference: bool| allocations_in(|| {
            std::hint::black_box(if reference {
                reference_signal_blocks(sig, fx.n)
            } else {
                SignalBlocks::build(sig, fx.n)
            });
        });
        let (r_short, r_long) = (count(&fx.short, true), count(&fx.long, true));
        assert!(
            r_long > r_short,
            "setup: the longer signal must run more transforms \
             ({r_short} vs {r_long} allocations in the original form)",
        );
        let (short, long) = (count(&fx.short, false), count(&fx.long, false));
        assert_eq!(
            short, long,
            "SignalBlocks::build allocates per transform: {short} allocations for the short \
             signal, {long} for one four times longer",
        );
    }

    /// The shared-route block loop makes the same allocations whatever the
    /// length.
    #[test]
    fn the_shared_route_does_not_allocate_per_transform() {
        let fx = ScratchFixture::new();
        let (bs, bl) = (SignalBlocks::build(&fx.short, fx.n), SignalBlocks::build(&fx.long, fx.n));
        let (fs, fl) = (frame_count(fx.short.len(), fx.hop), frame_count(fx.long.len(), fx.hop));
        assert!(bl.blocks() > bs.blocks(), "setup: blocks {} vs {}", bs.blocks(), bl.blocks());
        fx.m.complex_via_shared(&fx.short, &bs, fx.hop, fs).expect("shared route declined");
        reference_complex_via_shared(&fx.m, &fx.short, &bs, fx.hop, fs).expect("reference declined");

        let count = |sig: &[f32], blocks: &SignalBlocks, frames: usize, reference: bool| {
            allocations_in(|| {
                let out = if reference {
                    reference_complex_via_shared(&fx.m, sig, blocks, fx.hop, frames)
                } else {
                    fx.m.complex_via_shared(sig, blocks, fx.hop, frames)
                };
                assert!(std::hint::black_box(out).is_some(), "the shared route declined");
            })
        };
        let (r_short, r_long) = (count(&fx.short, &bs, fs, true), count(&fx.long, &bl, fl, true));
        assert!(
            r_long > r_short,
            "setup: the longer signal must run more transforms \
             ({r_short} vs {r_long} allocations in the original form)",
        );
        let (short, long) = (count(&fx.short, &bs, fs, false), count(&fx.long, &bl, fl, false));
        assert_eq!(
            short, long,
            "the shared route allocates per transform: {short} allocations over {} blocks, \
             {long} over {}",
            bs.blocks(), bl.blocks(),
        );
    }

    /// The own-transform block loop makes the same allocations whatever the
    /// length — both its transforms, forward and inverse.
    #[test]
    fn the_own_transform_route_does_not_allocate_per_transform() {
        let fx = ScratchFixture::new();
        let (fs, fl) = (frame_count(fx.short.len(), fx.hop), frame_count(fx.long.len(), fx.hop));
        fx.m.complex_via_fft(&fx.short, fx.hop, fs).expect("own-transform route declined");
        reference_complex_via_fft(&fx.m, &fx.short, fx.hop, fs).expect("reference declined");

        let count = |sig: &[f32], frames: usize, reference: bool| {
            allocations_in(|| {
                let out = if reference {
                    reference_complex_via_fft(&fx.m, sig, fx.hop, frames)
                } else {
                    fx.m.complex_via_fft(sig, fx.hop, frames)
                };
                assert!(std::hint::black_box(out).is_some(), "the own-transform route declined");
            })
        };
        let (r_short, r_long) = (count(&fx.short, fs, true), count(&fx.long, fl, true));
        assert!(
            r_long > r_short,
            "setup: the longer signal must run more transforms \
             ({r_short} vs {r_long} allocations in the original form)",
        );
        let (short, long) = (count(&fx.short, fs, false), count(&fx.long, fl, false));
        assert_eq!(
            short, long,
            "the own-transform route allocates per transform: {short} allocations for the \
             short signal, {long} for one four times longer",
        );
    }

    // ── Bounded bar-job scheduling (candidate G) ────────────────────────────

    fn bar_job_list(len: usize) -> Vec<BarJob> {
        (0..len).map(|bar| BarJob { bar, freq: 100.0 + bar as f32, q: 8.0 }).collect()
    }

    /// The production scheduling helper must hand a worker one job at a time.
    ///
    /// `fold` is how rayon exposes leaf boundaries: one accumulator per leaf. A
    /// single-worker pool removes stealing, so what remains is the splitter's own
    /// decision — exactly what `with_max_len(1)` constrains. Without that bound
    /// the same 64 jobs come back in fewer, larger leaves and this fails.
    #[test]
    fn the_bar_job_helper_splits_to_one_job_per_leaf() {
        use rayon::prelude::*;
        let jobs = bar_job_list(64);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("pool");
        let leaves: Vec<Vec<usize>> = pool.install(|| {
            bar_jobs(&jobs)
                .fold(Vec::new, |mut acc: Vec<usize>, job| {
                    acc.push(job.bar);
                    acc
                })
                .collect()
        });
        let sizes: Vec<usize> = leaves.iter().map(Vec::len).collect();
        println!("  {} leaves over {} jobs; sizes {sizes:?}", leaves.len(), jobs.len());
        assert_eq!(sizes.iter().sum::<usize>(), jobs.len(), "jobs went missing");
        assert!(
            sizes.iter().all(|&s| s == 1),
            "a leaf held more than one job, so one worker can still take the whole phase: {sizes:?}",
        );
        let mut seen: Vec<usize> = leaves.into_iter().flatten().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..jobs.len()).collect::<Vec<_>>(), "jobs duplicated or reordered away");
    }

    /// `install_columns` moves a finished column into its slot; it does not copy.
    ///
    /// **What this proves and what it does not.** It proves that for each of the
    /// three production sweeps' shared installation step, the heap allocation a
    /// worker filled is the allocation the slot holds afterwards — same pointer,
    /// so no deep copy was made. It is **not** a peak-memory measurement: it says
    /// nothing about how much memory the phase holds, and a run's peak is set by
    /// the columns and the transpose, not by this loop.
    ///
    /// It guards the production helper, not a copy of it. Replacing the move in
    /// `install_columns` with a clone fails this test; doing the same to an
    /// installation loop written out inside the test would not, which is why the
    /// helper exists.
    #[test]
    fn finished_columns_are_moved_into_their_slots() {
        use rayon::prelude::*;
        let jobs = bar_job_list(8);
        let built: Vec<(usize, Cols)> = bar_jobs(&jobs)
            .map(|job| {
                let mut col = Cols::zeroed(0, true);
                col.mix.extend((0..32).map(|i| i as f32));
                col.left.extend((0..32).map(|i| -(i as f32)));
                col.right.extend((0..32).map(|i| 2.0 * i as f32));
                (job.bar, col)
            })
            .collect();
        // Every plane, not only the mix: a joint column carries three.
        let ptrs: Vec<(usize, [*const f32; 3])> = built
            .iter()
            .map(|(bar, c)| (*bar, [c.mix.as_ptr(), c.left.as_ptr(), c.right.as_ptr()]))
            .collect();
        let values: Vec<(usize, Vec<f32>)> =
            built.iter().map(|(bar, c)| (*bar, c.mix.clone())).collect();

        let mut columns: Vec<Option<Cols>> = vec![None; jobs.len()];
        install_columns(&mut columns, built);

        for (bar, want) in ptrs {
            let c = columns[bar].as_ref().expect("installed");
            for (plane, (got, want)) in
                ["mix", "left", "right"].iter().zip([c.mix.as_ptr(), c.left.as_ptr(), c.right.as_ptr()].iter().zip(want.iter()))
            {
                assert_eq!(got, want, "bar {bar}'s {plane} plane was copied instead of moved");
            }
        }
        // And the move did not disturb the contents or the indexing.
        for (bar, want) in values {
            assert_eq!(columns[bar].as_ref().expect("installed").mix, want, "bar {bar} content");
        }
    }

    /// The transpose reorders faithfully and hands each column's memory back.
    ///
    /// Two separate claims. **Order:** `out[frame][bar]` is `columns[bar][frame]`
    /// for every cell, mono and joint, including a joint run's three planes and
    /// the awkward case of a column shorter than the frame count. **Release:**
    /// after a plane is transposed, every column of that plane has capacity
    /// zero — the allocation was returned, not merely emptied. `clear()` in
    /// place of `take` passes the first claim and fails the second, which is
    /// the whole point of the change.
    ///
    /// This bounds what is *live*, by construction, rather than measuring a
    /// process peak; the memory measurements are in the round's report.
    #[test]
    fn the_transpose_releases_each_column_as_it_consumes_it() {
        let (n_bars, frames) = (5usize, 7usize);
        let cell = |plane: usize, bar: usize, fi: usize| {
            (plane * 1_000 + bar * 100 + fi) as f32 + 0.5
        };

        let mut columns: Vec<Option<Cols>> = (0..n_bars)
            .map(|bar| {
                let mut c = Cols::zeroed(0, true);
                for fi in 0..frames {
                    c.mix.push(cell(0, bar, fi));
                    c.left.push(cell(1, bar, fi));
                    c.right.push(cell(2, bar, fi));
                }
                Some(c)
            })
            .collect();

        let mix = transpose_plane(&mut columns, frames, n_bars, |c| &mut c.mix);
        // The mix columns are gone; the other two planes are untouched.
        for (bar, slot) in columns.iter().enumerate() {
            let c = slot.as_ref().expect("slot");
            assert_eq!(c.mix.capacity(), 0, "bar {bar}: the mix column was emptied, not released");
            assert_eq!(c.left.len(), frames, "bar {bar}: the left column was disturbed");
            assert_eq!(c.right.len(), frames, "bar {bar}: the right column was disturbed");
        }
        let left = transpose_plane(&mut columns, frames, n_bars, |c| &mut c.left);
        let right = transpose_plane(&mut columns, frames, n_bars, |c| &mut c.right);
        for (bar, slot) in columns.iter().enumerate() {
            let c = slot.as_ref().expect("slot");
            for (name, cap) in [("left", c.left.capacity()), ("right", c.right.capacity())] {
                assert_eq!(cap, 0, "bar {bar}: the {name} column was emptied, not released");
            }
        }

        for (plane, out) in [(0usize, &mix), (1, &left), (2, &right)] {
            assert_eq!(out.len(), frames, "plane {plane}: wrong frame count");
            for (fi, row) in out.iter().enumerate() {
                assert_eq!(row.len(), n_bars, "plane {plane} frame {fi}: wrong bar count");
                for (bar, &got) in row.iter().enumerate() {
                    assert_eq!(
                        got.to_bits(), cell(plane, bar, fi).to_bits(),
                        "plane {plane} frame {fi} bar {bar}",
                    );
                }
            }
        }

        // A short column fills what it has; the rest of that bar stays zero.
        let mut ragged: Vec<Option<Cols>> = (0..n_bars)
            .map(|bar| {
                let mut c = Cols::zeroed(0, false);
                let have = if bar == 2 { frames - 3 } else { frames };
                for fi in 0..have {
                    c.mix.push(cell(0, bar, fi));
                }
                Some(c)
            })
            .collect();
        let out = transpose_plane(&mut ragged, frames, n_bars, |c| &mut c.mix);
        for (fi, row) in out.iter().enumerate() {
            for (bar, &got) in row.iter().enumerate() {
                let want = if bar == 2 && fi >= frames - 3 { 0.0 } else { cell(0, bar, fi) };
                assert_eq!(got.to_bits(), want.to_bits(), "ragged frame {fi} bar {bar}");
            }
        }
        for slot in &ragged {
            assert_eq!(slot.as_ref().expect("slot").mix.capacity(), 0);
        }
    }

    /// Every direct bar equals a sequential evaluation of the same superlet,
    /// bit for bit — mono and joint, including the derived mix.
    #[test]
    fn direct_bars_match_a_sequential_reference_bit_for_bit() {
        let sr = SR_A;
        let hop = hop_for_fps(sr, 60.0);
        let (lo, hi, n_bars) = (200.0f32, 5_000.0, 48);
        let c = cfg(1.0, 1.0, 4, 0.5);
        let bar_freq = |bar: usize| c.scale.bar_center(bar, n_bars, lo, hi);
        let bar_q = |bar: usize| {
            effective_q_at(bar_freq(bar), c.scale.grid_q_at(bar, n_bars, lo, hi), &c)
        };
        let direct: Vec<usize> = (0..n_bars)
            .filter(|&bar| !Superlet::new(bar_freq(bar), bar_q(bar), &c, sr as f32).prefers_fft(hop))
            .collect();
        assert!(direct.len() >= 8, "setup: only {} direct bars", direct.len());

        let work = WorkCells::default();
        let seams = Seams::production();

        // Mono.
        let sig = tone(&[300.0, 1_200.0], 0.5, sr);
        let frames = frame_count(sig.len(), hop);
        let got = analyze_via(never_gpu, &sig, sr, n_bars, lo, hi, hop, &c);
        assert_eq!(got.len(), frames, "setup: unexpected frame count");
        let chans = Input::Mono(&sig).chans();
        for &bar in &direct {
            let sl = Superlet::new(bar_freq(bar), bar_q(bar), &c, sr as f32);
            for (fi, row) in got.iter().enumerate() {
                let (m, _, _) = sl.frame(&chans, (fi * hop) as isize, &work, &seams);
                assert_eq!(
                    row[bar].to_bits(), m.to_bits(),
                    "mono bar {bar}, frame {fi}: {} vs {m}", row[bar],
                );
            }
        }

        // Joint: mix, left and right.
        let l = tone(&[300.0, 1_200.0], 0.5, sr);
        let r = tone(&[450.0, 2_000.0], 0.5, sr);
        let joint = analyze_joint_via(never_gpu, &l, &r, sr, n_bars, lo, hi, hop, &c);
        let jchans = Input::Joint { left: &l, right: &r }.chans();
        for &bar in &direct {
            let sl = Superlet::new(bar_freq(bar), bar_q(bar), &c, sr as f32);
            let rows = joint.mix.iter().zip(&joint.left).zip(&joint.right);
            for (fi, ((mix, left), right)) in rows.enumerate() {
                let (m, lv, rv) = sl.frame(&jchans, (fi * hop) as isize, &work, &seams);
                for (name, got_v, want) in [
                    ("mix", mix[bar], m),
                    ("left", left[bar], lv),
                    ("right", right[bar], rv),
                ] {
                    assert_eq!(
                        got_v.to_bits(), want.to_bits(),
                        "joint {name} bar {bar}, frame {fi}: {got_v} vs {want}",
                    );
                }
            }
        }
    }

    /// With every device dispatch failing, the missing-bar sweep must rebuild
    /// exactly what a CPU-only run produces — bit for bit, not to a tolerance:
    /// both go through the same shared-transform route on the cores.
    #[test]
    fn the_missing_bar_sweep_matches_a_cpu_only_run_bit_for_bit() {
        if super::super::gpu::GpuFft::shared().is_none() {
            println!("no GPU adapter — the forced-fallback sweep is not exercised here");
            return;
        }
        let sr = 16_000u32;
        let n = (3.0 * sr as f32) as usize;
        let tau = std::f32::consts::TAU;
        let l: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.5 * (tau * 45.0 * t).sin() + 0.2 * (tau * 900.0 * t).sin()
            })
            .collect();
        let r: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.4 * (tau * 45.0 * t + 0.7).sin() + 0.3 * (tau * 300.0 * t).sin()
            })
            .collect();
        let hop = hop_for_fps(sr, 60.0);
        let (lo, hi, bars) = (20.0f32, 8_000.0, 96);
        let c = cfg(1.0, 1.2, 4, 0.5);

        let every_dispatch_fails = |_chunk: usize, _channel: usize| true;
        let swept = analyze_joint_seamed(
            eligible_gpu,
            Seams { dispatch_fault: Some(&every_dispatch_fails), ..Seams::production() },
            &l, &r, sr, bars, lo, hi, hop, &c,
        );
        assert!(swept.work.device_failures > 0, "setup: no dispatch was made to fail");
        let cpu_only = analyze_joint_via(never_gpu, &l, &r, sr, bars, lo, hi, hop, &c);
        assert!(!cpu_only.mix.is_empty(), "setup: the CPU-only run produced nothing");

        for (name, a, b) in [
            ("mix", &swept.mix, &cpu_only.mix),
            ("left", &swept.left, &cpu_only.left),
            ("right", &swept.right, &cpu_only.right),
        ] {
            assert_eq!(a.len(), b.len(), "{name}: frame counts differ");
            for (fi, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
                for (bar, (x, y)) in ra.iter().zip(rb.iter()).enumerate() {
                    assert_eq!(
                        x.to_bits(), y.to_bits(),
                        "{name} frame {fi} bar {bar}: swept {x} vs CPU-only {y}",
                    );
                }
            }
        }
    }

    /// Progress reports every bar exactly once, and cancelling stops the run.
    ///
    /// **Portability.** The bound on how much work escapes a cancellation is a
    /// function of how many bars can be in flight, which is the worker count —
    /// so the test owns the worker count instead of inheriting the global pool,
    /// whose size is the machine's core count. Both cancellation cases run in an
    /// explicitly sized local pool and assert a bound derived from that size.
    /// Production cancellation is unchanged; only the test's control over
    /// in-flight work changed.
    #[test]
    fn progress_and_cancellation_survive_the_job_sweeps() {
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let sr = SR_A;
        let sig = tone(&[1_000.0], 0.5, sr);
        let hop = hop_for_fps(sr, 60.0);
        let (lo, hi, n_bars) = (200.0f32, 5_000.0, 64);
        let c = cfg(1.0, 1.0, 4, 0.5);

        const WORKERS: usize = 2;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(WORKERS)
            .build()
            .expect("local pool");

        // ── every bar reported exactly once on a complete run ─────────────
        let seen = Mutex::new(Vec::<usize>::new());
        let out = pool.install(|| {
            analyze_with_progress(
                &sig, sr, n_bars, lo, hi, hop, &c,
                &|| true,
                &|bar| seen.lock().unwrap_or_else(|e| e.into_inner()).push(bar),
            )
        });
        assert!(!out.is_empty(), "setup: the run produced nothing");
        let mut bars = seen.lock().unwrap_or_else(|e| e.into_inner()).clone();
        bars.sort_unstable();
        assert_eq!(
            bars, (0..n_bars).collect::<Vec<_>>(),
            "progress did not report every bar exactly once",
        );

        // ── cancelled between bars: bounded by the workers, not the machine ──
        //
        // Cancellation is polled before a bar is committed to, so a bar finishes
        // after the cut-off only if its check passed before it. At most one bar
        // per worker sits between "check passed" and "finished" at any instant,
        // so at most WORKERS more can complete after the LIMIT-th does.
        const LIMIT: usize = 8;
        let done = AtomicUsize::new(0);
        let cancelled = pool.install(|| {
            analyze_with_progress(
                &sig, sr, n_bars, lo, hi, hop, &c,
                &|| done.load(Ordering::Relaxed) < LIMIT,
                &|_| { done.fetch_add(1, Ordering::Relaxed); },
            )
        });
        assert!(cancelled.is_empty(), "a cancelled run must yield nothing");
        let ran = done.load(Ordering::Relaxed);
        assert!(
            ran <= LIMIT + WORKERS,
            "{ran} bars finished after cancelling at {LIMIT}, which {WORKERS} workers \
             cannot account for",
        );
        assert!(ran < n_bars, "the run was not cancelled at all: {ran} of {n_bars} bars");

        // ── cancelled *inside* a bar, deterministically ───────────────────
        //
        // The case above only ever reaches the frame-zero check, because its
        // bars are 30 frames long and the in-bar check fires every
        // CANCEL_CHECK_FRAMES. This one gives a single worker one long direct
        // bar and cancels on its third poll, which can only be the check at
        // frame 2 × CANCEL_CHECK_FRAMES.
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("pool");
        let long_hop = 8usize;
        let frames_wanted = 2 * CANCEL_CHECK_FRAMES + 1;
        let long_sig = tone(&[6_000.0], (frames_wanted * long_hop) as f32 / sr as f32, sr);
        let long_frames = frame_count(long_sig.len(), long_hop);
        assert!(
            long_frames > 2 * CANCEL_CHECK_FRAMES,
            "setup: {long_frames} frames is too few to reach the second in-bar check",
        );
        // One bar, high and narrow, so the run has no group phase at all and
        // every poll below belongs to the direct route. The condition is the
        // planner's own: a bar is direct when no member kernel is worth
        // transforming.
        let (dlo, dhi, d_bars) = (6_000.0f32, 9_000.0, 1);
        let dcfg = cfg(1.0, 1.0, 2, 0.5);
        for bar in 0..d_bars {
            let f = dcfg.scale.bar_center(bar, d_bars, dlo, dhi);
            let q = effective_q_at(f, dcfg.scale.grid_q_at(bar, d_bars, dlo, dhi), &dcfg);
            let transformed: Vec<usize> = superlet_cycles(q, &dcfg)
                .into_iter()
                .map(|c| morlet_taps(f, c, sr as f32))
                .filter(|&k| fft_is_cheaper(k, long_hop))
                .collect();
            assert!(
                transformed.is_empty() && f < sr as f32 / 2.0,
                "setup: bar {bar} at {f} Hz is not a direct bar: {transformed:?} taps transform",
            );
        }

        let polls = AtomicUsize::new(0);
        let finished = AtomicUsize::new(0);
        let stopped = one.install(|| {
            analyze_routed(
                Input::Mono(&long_sig), sr, d_bars, dlo, dhi, long_hop, &dcfg,
                &|| polls.fetch_add(1, Ordering::Relaxed) < 2,
                &|_| { finished.fetch_add(1, Ordering::Relaxed); },
                never_gpu,
                Seams::production(),
            )
            .mix
        });
        assert!(stopped.is_empty(), "a cancelled run must yield nothing");
        assert_eq!(
            finished.load(Ordering::Relaxed), 0,
            "the first bar completed, so the in-bar check never abandoned it",
        );
        assert_eq!(
            polls.load(Ordering::Relaxed), 3,
            "expected polls at frames 0, {CANCEL_CHECK_FRAMES} and {} of the first bar",
            2 * CANCEL_CHECK_FRAMES,
        );
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


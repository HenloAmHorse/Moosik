//! TEMPORARY offline study — is mono derivable from L/R spectra?
//!
//! Not part of the product. Delete when the v4 cache format is decided.
//!
//! Run:
//!   cargo test --release mono_study -- --ignored --nocapture --test-threads=1

use super::*;
use std::io::Write;

// The corrected storage experiment. Kept separate so the original study above
// stays exactly as it was run.
#[path = "aslt_mono_codec.rs"]
pub mod codec;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MUSIC_DIR: &str = "Test Music Files";
const OUT_DIR: &str = "target/mono_study";

// ---------------------------------------------------------------------------
// dB / cache-domain conversion — identical to `preprocess_file`
// ---------------------------------------------------------------------------

/// Cached value for a linear magnitude: `clamp((20·log10(mag·ref)+80)/80, 0, 1)`.
#[inline]
fn cache_v(mag: f32, db_ref: f32) -> f32 {
    let db = (20.0 * (mag * db_ref).log10()).max(-80.0);
    ((db + 80.0) / 80.0).clamp(0.0, 1.0)
}

/// Displayed dB for a linear magnitude, floored at −80 exactly as the cache is.
#[inline]
fn cache_db(mag: f32, db_ref: f32) -> f32 {
    cache_v(mag, db_ref) * 80.0 - 80.0
}

// ---------------------------------------------------------------------------
// Streaming error statistics
// ---------------------------------------------------------------------------

/// |Δ| histogram at 0.001 dB resolution up to 20 dB, then 0.1 dB to 200 dB.
struct Hist {
    fine: Vec<u64>,
    coarse: Vec<u64>,
    over: u64,
    n: u64,
    sum: f64,
    sum_sq: f64,
    max: f32,
    max_at: (usize, usize),
}

impl Hist {
    fn new() -> Self {
        Self {
            fine: vec![0; 20_000],
            coarse: vec![0; 1_800],
            over: 0,
            n: 0,
            sum: 0.0,
            sum_sq: 0.0,
            max: 0.0,
            max_at: (0, 0),
        }
    }

    fn push(&mut self, e: f32, frame: usize, bar: usize) {
        let a = e.abs();
        self.n += 1;
        self.sum += a as f64;
        self.sum_sq += (a as f64) * (a as f64);
        if a > self.max {
            self.max = a;
            self.max_at = (frame, bar);
        }
        if a < 20.0 {
            self.fine[(a * 1000.0) as usize] += 1;
        } else if a < 200.0 {
            self.coarse[((a - 20.0) * 10.0) as usize] += 1;
        } else {
            self.over += 1;
        }
    }

    fn pct(&self, p: f64) -> f32 {
        if self.n == 0 {
            return 0.0;
        }
        let target = (self.n as f64 * p).ceil() as u64;
        let mut c = 0u64;
        for (i, &v) in self.fine.iter().enumerate() {
            c += v;
            if c >= target {
                return i as f32 / 1000.0;
            }
        }
        for (i, &v) in self.coarse.iter().enumerate() {
            c += v;
            if c >= target {
                return 20.0 + i as f32 / 10.0;
            }
        }
        200.0
    }

    fn mean(&self) -> f32 {
        if self.n == 0 { 0.0 } else { (self.sum / self.n as f64) as f32 }
    }

    fn rms(&self) -> f32 {
        if self.n == 0 { 0.0 } else { (self.sum_sq / self.n as f64).sqrt() as f32 }
    }

    /// Fraction of cells whose error exceeds `t` dB.
    fn frac_over(&self, t: f32) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let mut c = 0u64;
        if t < 20.0 {
            let start = ((t * 1000.0) as usize).min(self.fine.len());
            c += self.fine[start..].iter().sum::<u64>();
            c += self.coarse.iter().sum::<u64>();
        } else {
            let start = (((t - 20.0) * 10.0) as usize).min(self.coarse.len());
            c += self.coarse[start..].iter().sum::<u64>();
        }
        c += self.over;
        c as f64 / self.n as f64
    }

    fn line(&self, label: &str) -> String {
        format!(
            "{label:<24} n={:<11} mean={:>8.4} rms={:>8.4} p50={:>7.3} p90={:>7.3} \
p99={:>7.3} p99.9={:>8.3} p99.99={:>8.3} max={:>9.3}@f{} b{}  >0.1dB={:>8.4}% >1dB={:>8.4}%",
            self.n,
            self.mean(),
            self.rms(),
            self.pct(0.50),
            self.pct(0.90),
            self.pct(0.99),
            self.pct(0.999),
            self.pct(0.9999),
            self.max,
            self.max_at.0,
            self.max_at.1,
            self.frac_over(0.1) * 100.0,
            self.frac_over(1.0) * 100.0,
        )
    }
}

/// One predictor, measured over all cells and over four visibility strata.
///
/// Two are absolute (the value is above a fixed dB line) and two are relative
/// to the loudest bar in the same frame. Absolute alone is not enough: the FFT
/// mappings normalise by window length and land 20 dB lower than the superlet
/// for identical audio, so a fixed line selects a different share of the
/// picture in each. Frame-relative selects "the part of this frame the eye is
/// actually on" in either.
struct Meter {
    name: String,
    all: Hist,
    abs70: Hist,
    abs40: Hist,
    rel40: Hist,
    rel20: Hist,
}

impl Meter {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            all: Hist::new(),
            abs70: Hist::new(),
            abs40: Hist::new(),
            rel40: Hist::new(),
            rel20: Hist::new(),
        }
    }

    /// `true_db` is absolute; `rel_db` is relative to the frame's loudest bar.
    fn push(&mut self, err_db: f32, true_db: f32, rel_db: f32, frame: usize, bar: usize) {
        self.all.push(err_db, frame, bar);
        if true_db > -70.0 {
            self.abs70.push(err_db, frame, bar);
        }
        if true_db > -40.0 {
            self.abs40.push(err_db, frame, bar);
        }
        if rel_db > -40.0 {
            self.rel40.push(err_db, frame, bar);
        }
        if rel_db > -20.0 {
            self.rel20.push(err_db, frame, bar);
        }
    }

    fn report(&self, w: &mut impl Write) {
        let _ = writeln!(w, "  {}", self.name);
        let _ = writeln!(w, "    {}", self.all.line("all cells"));
        let _ = writeln!(w, "    {}", self.abs70.line("abs > -70 dB"));
        let _ = writeln!(w, "    {}", self.abs40.line("abs > -40 dB"));
        let _ = writeln!(w, "    {}", self.rel40.line("within 40 dB of frame pk"));
        let _ = writeln!(w, "    {}", self.rel20.line("within 20 dB of frame pk"));
    }
}

// ---------------------------------------------------------------------------
// Bit packing — honest sizes for sub-byte planes
// ---------------------------------------------------------------------------

/// Pack `bits`-wide values (LSB-first) so a 4-bit plane costs half a byte per
/// cell rather than a whole one. Without this every sub-byte variant would be
/// measured at 8 bits and look pointlessly large.
fn pack_bits(vals: &[u8], bits: u32) -> Vec<u8> {
    if bits >= 8 {
        return vals.to_vec();
    }
    let mut out = Vec::with_capacity(vals.len() * bits as usize / 8 + 1);
    let mut acc: u32 = 0;
    let mut have: u32 = 0;
    for &v in vals {
        acc |= ((v as u32) & ((1 << bits) - 1)) << have;
        have += bits;
        while have >= 8 {
            out.push((acc & 0xFF) as u8);
            acc >>= 8;
            have -= 8;
        }
    }
    if have > 0 {
        out.push((acc & 0xFF) as u8);
    }
    out
}

/// mu-law compander over +-`range`, mapping to [-1, 1].
///
/// A uniform residual quantiser has to choose between clipping the rare large
/// correction and wasting most of its levels on values that never occur: the
/// distribution is a sharp spike at zero with a tail out to tens of dB.
/// Companding spends fine steps where the mass is and coarse ones in the tail,
/// which is the same reason telephony uses it.
fn compand(x: f32, range: f32, mu: f32) -> f32 {
    let a = (x.abs() / range).min(1.0);
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    s * (1.0 + mu * a).ln() / (1.0 + mu).ln()
}

/// Inverse of [`compand`].
fn expand(u: f32, range: f32, mu: f32) -> f32 {
    let a = u.abs().min(1.0);
    let s = if u < 0.0 { -1.0 } else { 1.0 };
    s * range * ((1.0 + mu).powf(a) - 1.0) / mu
}

/// Per-frequency-band error, so a result can say *where* it is wrong rather
/// than only how often.
struct Bands {
    edges: Vec<usize>,
    freqs: Vec<f32>,
    hist: Vec<Hist>,
}

impl Bands {
    fn new(n_bars: usize, min_freq: f32, max_freq: f32, scale: crate::spectrum::freq_scale::FreqScale) -> Self {
        const N: usize = 8;
        let edges: Vec<usize> = (0..=N).map(|i| i * n_bars / N).collect();
        let freqs: Vec<f32> = edges
            .iter()
            .map(|&b| scale.bar_center(b.min(n_bars - 1), n_bars, min_freq, max_freq))
            .collect();
        Self { edges, freqs, hist: (0..N).map(|_| Hist::new()).collect() }
    }

    fn band_of(&self, bar: usize) -> usize {
        match self.edges.binary_search(&bar) {
            Ok(i) => i.min(self.hist.len() - 1),
            Err(i) => (i - 1).min(self.hist.len() - 1),
        }
    }

    fn push(&mut self, err: f32, frame: usize, bar: usize) {
        let i = self.band_of(bar);
        self.hist[i].push(err, frame, bar);
    }

    fn report(&self, w: &mut impl Write, label: &str) {
        let _ = writeln!(w, "    {label} by band (dB error):");
        for (i, h) in self.hist.iter().enumerate() {
            let _ = writeln!(
                w,
                "      {:>7.0}-{:>7.0} Hz  n={:<10} mean={:>8.4} p90={:>7.3} p99={:>7.3} p99.9={:>8.3} max={:>9.3}",
                self.freqs[i],
                self.freqs[i + 1],
                h.n,
                h.mean(),
                h.pct(0.90),
                h.pct(0.99),
                h.pct(0.999),
                h.max,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Decode to separate L/R at native rate, exactly as `preprocess_file` reads
/// samples (rodio i16 → /32768), but keeping the channels apart.
fn decode_stereo(path: &Path) -> Result<(Vec<f32>, Vec<f32>, u32), String> {
    use rodio::Decoder;
    use rodio::Source;
    use std::io::BufReader;

    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let decoder = Decoder::new(BufReader::new(file)).map_err(|e| e.to_string())?;
    let sr = decoder.sample_rate();
    let ch = (decoder.channels() as usize).max(1);
    if ch != 2 {
        return Err(format!("{ch} channels, need 2"));
    }
    let hint = decoder
        .total_duration()
        .map(|d| (d.as_secs_f64() * sr as f64) as usize)
        .unwrap_or(0);

    let mut l = Vec::with_capacity(hint.max(1));
    let mut r = Vec::with_capacity(hint.max(1));
    let mut it = decoder;
    loop {
        match (it.next(), it.next()) {
            (Some(a), Some(b)) => {
                l.push(a as f32 / 32_768.0);
                r.push(b as f32 / 32_768.0);
            }
            _ => break,
        }
    }
    Ok((l, r, sr))
}

// ---------------------------------------------------------------------------
// Analysis fronts
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mapping {
    Superlet,
    Cqt,
    Gaussian,
    FlatOverlap,
}

impl Mapping {
    fn name(self) -> &'static str {
        match self {
            Mapping::Superlet => "Superlet",
            Mapping::Cqt => "CQT",
            Mapping::Gaussian => "Gaussian",
            Mapping::FlatOverlap => "FlatOverlap",
        }
    }
    fn as_mode(self) -> crate::spectrum::BarMappingMode {
        match self {
            Mapping::Superlet => crate::spectrum::BarMappingMode::Superlet,
            Mapping::Cqt => crate::spectrum::BarMappingMode::Cqt,
            Mapping::Gaussian => crate::spectrum::BarMappingMode::Gaussian,
            Mapping::FlatOverlap => crate::spectrum::BarMappingMode::FlatOverlap,
        }
    }
}

struct Params {
    n_bars: usize,
    min_freq: f32,
    max_freq: f32,
    pre_fps: f32,
    aslt_cfg: AsltConfig,
    pad_factor: usize,
    overlap: f32,
    window_fn: crate::spectrum::WindowFn,
    interp: crate::spectrum::InterpolationMode,
}

/// The FFT bar mapping, copied verbatim from `preprocess_file`'s phase-2 loop
/// so the study measures what the product caches, not an approximation.
fn fft_analyze(
    signal: &[f32],
    sample_rate: u32,
    p: &Params,
    mapping: Mapping,
    fft_size: usize,
    hop: usize,
) -> Vec<Vec<f32>> {
    use rayon::prelude::*;
    use rustfft::{FftPlanner, num_complex::Complex};

    if signal.len() < fft_size {
        return Vec::new();
    }
    let padded_size = fft_size * p.pad_factor;
    let window = crate::spectrum::make_window(fft_size, &p.window_fn);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(padded_size);
    let half = padded_size / 2;
    let log_min = p.min_freq.log10();
    let log_max = (sample_rate as f32 / 2.0).min(p.max_freq).log10();
    let n_bars = p.n_bars;
    let bar_mapping = mapping.as_mode();

    let num_frames = (signal.len() - fft_size) / hop + 1;
    let n_octaves_pp = (log_max - log_min) / 2_f32.log10();
    let bins_per_oct_pp = n_bars as f32 / n_octaves_pp.max(0.1);
    let cqt_q_pp = 1.0 / (2.0_f32.powf(1.0 / bins_per_oct_pp) - 1.0);

    (0..num_frames)
        .into_par_iter()
        .map(|fi| {
            let start = fi * hop;
            let slice = &signal[start..start + fft_size];
            let mut buf: Vec<Complex<f32>> = slice
                .iter()
                .zip(window.iter())
                .map(|(s, w)| Complex { re: s * w, im: 0.0 })
                .chain(std::iter::repeat_n(
                    Complex { re: 0.0, im: 0.0 },
                    padded_size - fft_size,
                ))
                .collect();
            fft.process(&mut buf);
            let norms: Vec<f32> = buf[..half].iter().map(|c: &Complex<f32>| c.norm()).collect();

            (0..n_bars)
                .map(|bar| {
                    let fscale = p.aslt_cfg.scale;
                    if bar_mapping == crate::spectrum::BarMappingMode::Cqt {
                        let tc = (bar as f32 + 0.5) / n_bars as f32;
                        let f_c = fscale.freq_at(tc, p.min_freq, p.max_freq);
                        let bc = (f_c * padded_size as f32 / sample_rate as f32)
                            .clamp(1.0, half as f32 - 1.0);
                        crate::spectrum::cqt_kernel(&norms, bc, cqt_q_pp, half)
                    } else {
                        let t0 = bar as f32 / n_bars as f32;
                        let t1 = (bar + 1) as f32 / n_bars as f32;
                        let freq_lo = fscale.freq_at(t0, p.min_freq, p.max_freq);
                        let freq_hi = fscale.freq_at(t1, p.min_freq, p.max_freq);
                        let fbin_lo =
                            (freq_lo * padded_size as f32 / sample_rate as f32).max(1.0);
                        let fbin_hi = (freq_hi * padded_size as f32 / sample_rate as f32)
                            .max(fbin_lo + 0.001)
                            .min(half as f32 - 0.001);
                        if fbin_hi - fbin_lo <= 1.0 {
                            let center = (fbin_lo + fbin_hi) * 0.5;
                            crate::spectrum::interp_sub_bin(&norms, center, &p.interp)
                        } else {
                            let b_start = fbin_lo.floor() as usize;
                            let b_end = (fbin_hi.ceil() as usize).min(half - 1);
                            let bc = (fbin_lo + fbin_hi) * 0.5;
                            let sigma = ((fbin_hi - fbin_lo) * 0.5).max(0.5);
                            let mut wsum = 0.0_f32;
                            let mut weight = 0.0_f32;
                            for (b_idx, &norm_b) in norms[b_start..=b_end].iter().enumerate() {
                                let b = b_start + b_idx;
                                let w = match bar_mapping {
                                    crate::spectrum::BarMappingMode::FlatOverlap => {
                                        (fbin_hi.min(b as f32 + 1.0) - fbin_lo.max(b as f32))
                                            .max(0.0)
                                    }
                                    _ => {
                                        let center_b = b as f32 + 0.5;
                                        (-(center_b - bc).powi(2) / (2.0 * sigma * sigma)).exp()
                                    }
                                };
                                wsum += norm_b * w;
                                weight += w;
                            }
                            if weight > 0.0 { wsum / weight } else { 0.0 }
                        }
                    }
                })
                .collect()
        })
        .collect()
}

/// Run one mapping over one signal, returning raw linear magnitudes and the
/// dB reference the product would apply to them.
fn run(signal: &[f32], sample_rate: u32, p: &Params, mapping: Mapping) -> (Vec<Vec<f32>>, f32) {
    match mapping {
        Mapping::Superlet => {
            let hop = hop_for_fps(sample_rate, p.pre_fps);
            let a_max = (sample_rate as f32 / 2.0).min(p.max_freq);
            let out = crate::spectrum::gpu_calib::install(|| {
                analyze(signal, sample_rate, p.n_bars, p.min_freq, a_max, hop, &p.aslt_cfg)
            });
            (out, crate::spectrum::ASLT_DB_REF)
        }
        _ => {
            let fft_size = crate::spectrum::auto_fft_size_for(sample_rate);
            let hop = ((fft_size as f32 * (1.0 - p.overlap)).round() as usize).max(1);
            let out = fft_analyze(signal, sample_rate, p, mapping, fft_size, hop);
            (out, 1.0 / fft_size as f32)
        }
    }
}

// ---------------------------------------------------------------------------
// Compression probes — the real v3 encoder for u16 planes
// ---------------------------------------------------------------------------

fn v3_bytes(frames: &[Vec<f32>], tag: &str) -> u64 {
    let p = PathBuf::from(OUT_DIR).join(format!("probe_{tag}.bin"));
    crate::spectrum::save_cache(&p, frames);
    let n = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
    let _ = std::fs::remove_file(&p);
    n
}

/// Compressed size of a `bits`-wide plane, both raw and zigzag-delta across
/// frequency — the same two forms v3 chooses between for its u16 planes. The
/// delta is wrapped into a signed `bits`-wide value before zigzagging, exactly
/// as v3 wraps into `i16`, so it still reconstructs exactly while keeping the
/// common small steps small.
fn plane_bytes(vals: &[u8], bits: u32, n_bars: usize) -> (u64, u64) {
    let raw = lz4_flex::compress_prepend_size(&pack_bits(vals, bits)).len() as u64;
    let shift = 32 - bits;
    let mask = if bits >= 8 { 0xFFu32 } else { (1u32 << bits) - 1 };
    let mut d = Vec::with_capacity(vals.len());
    for row in vals.chunks(n_bars.max(1)) {
        let mut prev = 0i32;
        for &v in row {
            let diff = ((v as i32 - prev) << shift) >> shift;
            prev = v as i32;
            let z = (((diff << 1) ^ (diff >> 31)) as u32) & mask;
            d.push(z as u8);
        }
    }
    let delta = lz4_flex::compress_prepend_size(&pack_bits(&d, bits)).len() as u64;
    (raw, delta)
}

// ---------------------------------------------------------------------------
// The study
// ---------------------------------------------------------------------------

fn mb(b: u64) -> f64 {
    b as f64 / 1_048_576.0
}

fn study_track(path: &Path, p: &Params, mappings: &[Mapping], w: &mut impl Write) {
    let name = path.file_name().unwrap().to_string_lossy().to_string();
    let t0 = Instant::now();
    let (l_sig, r_sig, sr) = match decode_stereo(path) {
        Ok(v) => v,
        Err(e) => {
            let _ = writeln!(w, "\n### {name}\n  SKIPPED: {e}");
            let _ = w.flush();
            return;
        }
    };
    let m_sig: Vec<f32> = l_sig
        .iter()
        .zip(&r_sig)
        .map(|(&a, &b)| (a + b) / 2.0)
        .collect();

    // Correlation of the raw signals — the one-number description of how
    // stereo this track is, so the per-track results can be read against it.
    let (mut sll, mut srr, mut slr) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in l_sig.iter().zip(&r_sig) {
        sll += (a * a) as f64;
        srr += (b * b) as f64;
        slr += (a * b) as f64;
    }
    let corr = if sll > 0.0 && srr > 0.0 {
        slr / (sll * srr).sqrt()
    } else {
        1.0
    };
    let side_db = {
        let mut ss = 0.0f64;
        for (&a, &b) in l_sig.iter().zip(&r_sig) {
            let s = ((a - b) / 2.0) as f64;
            ss += s * s;
        }
        let mid = (sll + srr + 2.0 * slr) / 4.0;
        if mid > 0.0 && ss > 0.0 {
            10.0 * (ss / mid).log10()
        } else {
            -120.0
        }
    };

    let _ = writeln!(
        w,
        "\n\n### {name}\n  {sr} Hz, {:.1} s, {} samples/ch, decode {:.1} s\n  \
L/R correlation {corr:+.4},  side/mid {side_db:+.2} dB",
        l_sig.len() as f64 / sr as f64,
        l_sig.len(),
        t0.elapsed().as_secs_f64(),
    );
    let _ = w.flush();

    for &mapping in mappings {
        let ta = Instant::now();
        let (la, db_ref) = run(&l_sig, sr, p, mapping);
        let t_l = ta.elapsed().as_secs_f64();
        let tb = Instant::now();
        let (ra, _) = run(&r_sig, sr, p, mapping);
        let t_r = tb.elapsed().as_secs_f64();
        let tc = Instant::now();
        let (ma, _) = run(&m_sig, sr, p, mapping);
        let t_m = tc.elapsed().as_secs_f64();

        if la.is_empty() || ra.is_empty() || ma.is_empty() {
            let _ = writeln!(w, "  {} — analysis returned nothing", mapping.name());
            continue;
        }
        let frames = la.len().min(ra.len()).min(ma.len());
        let bars = la[0].len();

        let _ = writeln!(
            w,
            "\n  --- {} --- {frames} frames x {bars} bars   \
analysis L {t_l:.1}s  R {t_r:.1}s  M {t_m:.1}s",
            mapping.name()
        );
        eprintln!(
            "    {} done: {frames}x{bars}, L {t_l:.1}s R {t_r:.1}s M {t_m:.1}s",
            mapping.name()
        );

        // Residual candidates: how wide a window either side of the P0
        // prediction the correction covers, and at how many bits. A narrow
        // window quantises finely but clips the rare large corrections; a wide
        // one never clips but wastes most of its levels on values that never
        // occur. Which trade wins is the whole question, so measure the grid.
        const RES_SPECS: [(f32, u32); 8] = [
            (80.0, 8),
            (40.0, 8),
            (20.0, 8),
            (12.0, 8),
            (20.0, 6),
            (12.0, 6),
            (12.0, 4),
            (6.0, 4),
        ];
        // Companded residual: (range dB, bits, mu).
        const CRES_SPECS: [(f32, u32, f32); 8] = [
            (80.0, 8, 255.0),
            (80.0, 7, 255.0),
            (80.0, 6, 255.0),
            (80.0, 5, 255.0),
            (80.0, 8, 1023.0),
            (80.0, 6, 1023.0),
            (80.0, 8, 63.0),
            (80.0, 6, 63.0),
        ];
        const COS_SPECS: [u32; 3] = [8, 6, 4];
        // Index into RES_SPECS whose per-band breakdown is reported.
        const RES_BAND_IDX: usize = 3;

        let mut m_sum = Meter::new(
            "P0  arithmetic mean (|L|+|R|)/2        [no phase, in-phase assumption]",
        );
        let mut m_rms = Meter::new(
            "P1  power mean sqrt(|L|^2+|R|^2)/2     [no phase, uncorrelated assumption]",
        );
        let mut m_max =
            Meter::new("P2  louder channel max(|L|,|R|)/2      [reference floor]");
        let mut m_orc = Meter::new(
            "P3  ORACLE cos-phi, clamped to [-1,1]  [ceiling of any 1-scalar format]",
        );
        let mut m_oor = Meter::new(
            "P3b ORACLE restricted to the cells where it had to clamp [the real cost]",
        );
        let mut m_q8 = Meter::new("P4  oracle cos-phi at 8 bits");
        let mut m_q6 = Meter::new("P5  oracle cos-phi at 6 bits");
        let mut m_q4 = Meter::new("P6  oracle cos-phi at 4 bits");
        let mut res_meters: Vec<Meter> = RES_SPECS
            .iter()
            .map(|(r, b)| {
                Meter::new(&format!(
                    "R{b}@+-{r:.0}dB  P0 + quantised residual  [step {:.4} dB]",
                    2.0 * r / (((1u32 << b) - 1) as f32)
                ))
            })
            .collect();

        let mut cres_meters: Vec<Meter> = CRES_SPECS
            .iter()
            .map(|(r, b, mu)| {
                let levels = ((1u32 << b) - 1) as f32;
                // Step at zero, which is where almost every cell sits.
                let fine = expand(2.0 / levels, *r, *mu);
                Meter::new(&format!(
                    "M{b}@+-{r:.0}dB mu={mu:.0}  P0 + companded residual  [step at 0: {fine:.4} dB]"
                ))
            })
            .collect();
        let mut cres_planes: Vec<Vec<u8>> = CRES_SPECS
            .iter()
            .map(|_| Vec::with_capacity(frames * bars))
            .collect();
        let mut res_planes: Vec<Vec<u8>> = RES_SPECS
            .iter()
            .map(|_| Vec::with_capacity(frames * bars))
            .collect();
        let mut cos_planes: Vec<Vec<u8>> = COS_SPECS
            .iter()
            .map(|_| Vec::with_capacity(frames * bars))
            .collect();
        let mut plane_res_u16: Vec<Vec<f32>> = Vec::with_capacity(frames);

        let sc = p.aslt_cfg.scale;
        let mut bands_p0 = Bands::new(bars, p.min_freq, p.max_freq, sc);
        let mut bands_cos8 = Bands::new(bars, p.min_freq, p.max_freq, sc);
        let mut bands_res = Bands::new(bars, p.min_freq, p.max_freq, sc);
        let mut bands_oor = Bands::new(bars, p.min_freq, p.max_freq, sc);

        let mut out_hi = 0u64;
        let mut out_lo = 0u64;
        let mut total = 0u64;
        let mut worst_hi = 1.0f32;
        let mut worst_lo = -1.0f32;
        let mut res_max_db = 0.0f32;
        // How much of the residual would clip at each candidate window.
        let mut clip_counts = [0u64; RES_SPECS.len()];

        for f in 0..frames {
            let (lr, rr, mr) = (&la[f], &ra[f], &ma[f]);
            // Frame peak, for the relative strata. Computed on true mono, so
            // every predictor is judged against the same reference.
            let frame_max_db = mr
                .iter()
                .fold(-80.0f32, |a, &m| a.max(cache_db(m, db_ref)));
            let mut res_row = Vec::with_capacity(bars);

            for b in 0..bars {
                let (lv, rv, mv) = (lr[b], rr[b], mr[b]);
                let true_db = cache_db(mv, db_ref);
                let rel_db = true_db - frame_max_db;
                total += 1;

                let p0 = (lv + rv) / 2.0;
                let p1 = (lv * lv + rv * rv).max(0.0).sqrt() / 2.0;
                let p2 = lv.max(rv) / 2.0;
                let p0_db = cache_db(p0, db_ref);
                m_sum.push(p0_db - true_db, true_db, rel_db, f, b);
                m_rms.push(cache_db(p1, db_ref) - true_db, true_db, rel_db, f, b);
                m_max.push(cache_db(p2, db_ref) - true_db, true_db, rel_db, f, b);
                bands_p0.push(p0_db - true_db, f, b);

                // Oracle: the cos(phi) that makes |M| exact, if one exists.
                let denom = 2.0 * lv * rv;
                let raw_cos = if denom > 0.0 {
                    (4.0 * mv * mv - lv * lv - rv * rv) / denom
                } else {
                    // One channel silent: mono is exactly half the other and
                    // the phase term vanishes, so any value reproduces it.
                    0.0
                };
                let oor = raw_cos.is_finite() && !(-1.0..=1.0).contains(&raw_cos);
                if raw_cos.is_finite() {
                    if raw_cos > 1.0 {
                        out_hi += 1;
                        worst_hi = worst_hi.max(raw_cos);
                    } else if raw_cos < -1.0 {
                        out_lo += 1;
                        worst_lo = worst_lo.min(raw_cos);
                    }
                }
                let c = if raw_cos.is_finite() {
                    raw_cos.clamp(-1.0, 1.0)
                } else {
                    0.0
                };

                let from_cos = |cc: f32| {
                    (lv * lv + rv * rv + 2.0 * lv * rv * cc).max(0.0).sqrt() / 2.0
                };
                let orc_err = cache_db(from_cos(c), db_ref) - true_db;
                m_orc.push(orc_err, true_db, rel_db, f, b);
                if oor {
                    m_oor.push(orc_err, true_db, rel_db, f, b);
                    bands_oor.push(orc_err, f, b);
                }

                let dq = |bits: u32| -> f32 {
                    let levels = ((1u32 << bits) - 1) as f32;
                    let qi = ((c + 1.0) * 0.5 * levels).round().clamp(0.0, levels);
                    qi / levels * 2.0 - 1.0
                };
                let e8 = cache_db(from_cos(dq(8)), db_ref) - true_db;
                m_q8.push(e8, true_db, rel_db, f, b);
                m_q6.push(cache_db(from_cos(dq(6)), db_ref) - true_db, true_db, rel_db, f, b);
                m_q4.push(cache_db(from_cos(dq(4)), db_ref) - true_db, true_db, rel_db, f, b);
                bands_cos8.push(e8, f, b);

                for (i, &bits) in COS_SPECS.iter().enumerate() {
                    let levels = ((1u32 << bits) - 1) as f32;
                    cos_planes[i]
                        .push(((c + 1.0) * 0.5 * levels).round().clamp(0.0, levels) as u8);
                }

                // Residual coding: store what P0 got wrong, not mono itself.
                let res_db = true_db - p0_db;
                res_max_db = res_max_db.max(res_db.abs());
                for (i, &(range, bits)) in RES_SPECS.iter().enumerate() {
                    if res_db.abs() > range {
                        clip_counts[i] += 1;
                    }
                    let levels = ((1u32 << bits) - 1) as f32;
                    let step = 2.0 * range / levels;
                    let qi = ((res_db.clamp(-range, range) + range) / step)
                        .round()
                        .clamp(0.0, levels);
                    let rec_db = (p0_db + (qi * step - range)).clamp(-80.0, 0.0);
                    let err = rec_db - true_db;
                    res_meters[i].push(err, true_db, rel_db, f, b);
                    if i == RES_BAND_IDX {
                        bands_res.push(err, f, b);
                    }
                    res_planes[i].push(qi as u8);
                }

                for (i, &(range, bits, mu)) in CRES_SPECS.iter().enumerate() {
                    let levels = ((1u32 << bits) - 1) as f32;
                    let u = compand(res_db, range, mu);
                    let qi = ((u + 1.0) * 0.5 * levels).round().clamp(0.0, levels);
                    let rec_db =
                        (p0_db + expand(qi / levels * 2.0 - 1.0, range, mu)).clamp(-80.0, 0.0);
                    cres_meters[i].push(rec_db - true_db, true_db, rel_db, f, b);
                    cres_planes[i].push(qi as u8);
                }

                // The same residual as a u16 plane through the real v3 coder,
                // for a like-for-like size against the magnitude planes.
                let res_v = cache_v(mv, db_ref) - cache_v(p0, db_ref);
                res_row.push(res_v * 0.5 + 0.5);
            }
            plane_res_u16.push(res_row);
        }

        let _ = writeln!(
            w,
            "    oracle cos out of range: >1 {:.6}% (worst {worst_hi:.4})  \
<-1 {:.6}% (worst {worst_lo:.4})   |residual|max {res_max_db:.2} dB",
            out_hi as f64 / total as f64 * 100.0,
            out_lo as f64 / total as f64 * 100.0,
        );
        for m in [&m_sum, &m_rms, &m_max, &m_orc, &m_oor, &m_q8, &m_q6, &m_q4] {
            m.report(w);
        }
        for (i, m) in res_meters.iter().enumerate() {
            let _ = writeln!(
                w,
                "  [clips {:.6}% of cells]",
                clip_counts[i] as f64 / total as f64 * 100.0
            );
            m.report(w);
        }
        for m in cres_meters.iter() {
            m.report(w);
        }
        bands_p0.report(w, "P0 no-phase");
        bands_cos8.report(w, "cos-phi 8-bit");
        bands_res.report(
            w,
            &format!(
                "residual {}b@+-{:.0}dB",
                RES_SPECS[RES_BAND_IDX].1, RES_SPECS[RES_BAND_IDX].0
            ),
        );
        bands_oor.report(w, "oracle-clamped cells only");

        // Sizes, through the v3 coder the product itself uses for u16 planes.
        let to_v = |a: &Vec<Vec<f32>>| -> Vec<Vec<f32>> {
            a.iter()
                .take(frames)
                .map(|row| row.iter().map(|&m| cache_v(m, db_ref)).collect())
                .collect()
        };
        let b_m = v3_bytes(&to_v(&ma), "m");
        let b_l = v3_bytes(&to_v(&la), "l");
        let b_r = v3_bytes(&to_v(&ra), "r");
        let b_res16 = v3_bytes(&plane_res_u16, "res");
        drop(plane_res_u16);

        let _ = writeln!(w, "\n    plane sizes (MB)");
        let _ = writeln!(
            w,
            "      mono u16-v3 {:.2}   L u16-v3 {:.2}   R u16-v3 {:.2}   residual u16-v3 {:.2}",
            mb(b_m),
            mb(b_l),
            mb(b_r),
            mb(b_res16)
        );
        let mut cos_sizes = Vec::new();
        for (i, &bits) in COS_SPECS.iter().enumerate() {
            let (raw, delta) = plane_bytes(&cos_planes[i], bits, bars);
            cos_sizes.push(raw.min(delta));
            let _ = writeln!(
                w,
                "      cos-phi {bits}b {:.2}  (raw {:.2} / delta {:.2})",
                mb(raw.min(delta)),
                mb(raw),
                mb(delta)
            );
        }
        let mut cres_sizes = Vec::new();
        for (i, &(range, bits, mu)) in CRES_SPECS.iter().enumerate() {
            let (raw, delta) = plane_bytes(&cres_planes[i], bits, bars);
            cres_sizes.push(raw.min(delta));
            let _ = writeln!(
                w,
                "      companded {bits}b@+-{range:.0}dB mu={mu:.0} {:.2}  (raw {:.2} / delta {:.2})",
                mb(raw.min(delta)),
                mb(raw),
                mb(delta)
            );
        }
        let mut res_sizes = Vec::new();
        for (i, &(range, bits)) in RES_SPECS.iter().enumerate() {
            let (raw, delta) = plane_bytes(&res_planes[i], bits, bars);
            res_sizes.push(raw.min(delta));
            let _ = writeln!(
                w,
                "      residual {bits}b@+-{range:.0}dB {:.2}  (raw {:.2} / delta {:.2})",
                mb(raw.min(delta)),
                mb(raw),
                mb(delta)
            );
        }

        let _ = writeln!(w, "\n    format totals (MB)");
        let _ = writeln!(w, "      mono only, today                {:.2}", mb(b_m));
        let _ = writeln!(w, "      A: L+R, no mono                 {:.2}", mb(b_l + b_r));
        for (i, &bits) in COS_SPECS.iter().enumerate() {
            let _ = writeln!(
                w,
                "      B{bits}: L+R+cos-phi {bits}b            {:.2}   (+{:.1}% over mono-only)",
                mb(b_l + b_r + cos_sizes[i]),
                (b_l + b_r + cos_sizes[i]) as f64 / b_m as f64 * 100.0 - 100.0
            );
        }
        for (i, &(range, bits)) in RES_SPECS.iter().enumerate() {
            let _ = writeln!(
                w,
                "      C: L+R+residual {bits}b@+-{range:.0}dB    {:.2}   (+{:.1}% over mono-only)",
                mb(b_l + b_r + res_sizes[i]),
                (b_l + b_r + res_sizes[i]) as f64 / b_m as f64 * 100.0 - 100.0
            );
        }
        for (i, &(range, bits, mu)) in CRES_SPECS.iter().enumerate() {
            let _ = writeln!(
                w,
                "      M: L+R+companded {bits}b@+-{range:.0}dB mu={mu:.0}  {:.2}   (+{:.1}% over mono-only)",
                mb(b_l + b_r + cres_sizes[i]),
                (b_l + b_r + cres_sizes[i]) as f64 / b_m as f64 * 100.0 - 100.0
            );
        }
        let _ = writeln!(
            w,
            "      D: L+R+M, two caches            {:.2}   (+{:.1}% over mono-only)",
            mb(b_m + b_l + b_r),
            (b_m + b_l + b_r) as f64 / b_m as f64 * 100.0 - 100.0
        );
        let _ = w.flush();
    }
}

#[test]
#[ignore]
fn mono_study() {
    let _ = std::fs::create_dir_all(OUT_DIR);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let out = PathBuf::from(OUT_DIR).join(format!("report_{stamp}.txt"));
    let mut w = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    eprintln!("report -> {}", out.display());

    // The owner's own settings, read from disk so the study measures what they
    // actually run rather than the shipping defaults.
    let s = crate::spectrum::load_spectrum_settings().expect("no ~/.moosik/spectrum.json");
    let p = Params {
        n_bars: s.bar_count,
        min_freq: s.min_freq,
        max_freq: s.max_freq,
        pre_fps: s.pre_fps,
        aslt_cfg: s.aslt_cfg.clone(),
        pad_factor: s.pad_factor,
        overlap: s.overlap,
        window_fn: s.window_fn.clone(),
        interp: s.interp_mode.clone(),
    };

    let _ = writeln!(
        w,
        "mono-derivation study\nbars {} | {:.0}-{:.0} Hz | pre_fps {} | pad {} | \
overlap {} | aslt {:?}",
        p.n_bars, p.min_freq, p.max_freq, p.pre_fps, p.pad_factor, p.overlap, p.aslt_cfg,
    );

    let only: Option<String> = std::env::var("STUDY_ONLY").ok();
    let maps: Vec<Mapping> = match std::env::var("STUDY_MAPPINGS").as_deref() {
        Ok("superlet") => vec![Mapping::Superlet],
        Ok("fft") => vec![Mapping::Cqt, Mapping::Gaussian, Mapping::FlatOverlap],
        Ok("cqt") => vec![Mapping::Cqt],
        _ => vec![
            Mapping::Superlet,
            Mapping::Cqt,
            Mapping::Gaussian,
            Mapping::FlatOverlap,
        ],
    };

    let mut files: Vec<PathBuf> = std::fs::read_dir(MUSIC_DIR)
        .expect("Test Music Files")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    files.sort();

    for f in files {
        if let Some(o) = &only
            && !f.to_string_lossy().contains(o.as_str())
        {
            continue;
        }
        eprintln!("=== {}", f.display());
        study_track(&f, &p, &maps, &mut w);
    }
    let _ = w.flush();
    eprintln!("done -> {}", out.display());
}

// ---------------------------------------------------------------------------
// Bit-depth sweep — how fine does a cached value actually need to be?
// ---------------------------------------------------------------------------

/// Pack `bits`-wide values (LSB first) for widths up to 16.
fn pack_bits16(vals: &[u16], bits: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * bits as usize / 8 + 2);
    let mut acc: u32 = 0;
    let mut have: u32 = 0;
    let mask: u32 = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
    for &v in vals {
        acc |= (v as u32 & mask) << have;
        have += bits;
        while have >= 8 {
            out.push((acc & 0xFF) as u8);
            acc >>= 8;
            have -= 8;
        }
    }
    if have > 0 {
        out.push((acc & 0xFF) as u8);
    }
    out
}

/// Bit-packed plane, raw and zigzag-delta across frequency, whichever is
/// smaller. The delta is wrapped into a signed `bits`-wide value before
/// zigzagging, exactly as v3 wraps into `i16`.
fn packed_plane_bytes(vals: &[u16], bits: u32, n_bars: usize) -> u64 {
    let raw = lz4_flex::compress_prepend_size(&pack_bits16(vals, bits)).len() as u64;
    let shift = 32 - bits;
    let mask: u32 = (1u32 << bits) - 1;
    let mut d = Vec::with_capacity(vals.len());
    for row in vals.chunks(n_bars.max(1)) {
        let mut prev = 0i32;
        for &v in row {
            let diff = ((v as i32 - prev) << shift) >> shift;
            prev = v as i32;
            d.push(((((diff << 1) ^ (diff >> 31)) as u32) & mask) as u16);
        }
    }
    let delta = lz4_flex::compress_prepend_size(&pack_bits16(&d, bits)).len() as u64;
    raw.min(delta)
}

/// Requantise onto a `2^bits - 1` grid and return it in the [0, 1] domain.
///
/// This is the obvious way to write it and the wrong one for the shipping
/// coder: rescaled by 65535 the values become multiples of 257, not 256, so the
/// low byte still cycles through every value and the v3 byte-plane split has
/// nothing to exploit.
#[inline]
fn requantise_unaligned(v: f32, bits: u32) -> f32 {
    let levels = ((1u32 << bits) - 1) as f32;
    (v.clamp(0.0, 1.0) * levels).round() / levels
}

/// Requantise by zeroing the low `16 - bits` bits of the u16 the coder will
/// store, which is the version that lets the existing byte-plane split work.
#[inline]
fn requantise_aligned(v: f32, bits: u32) -> f32 {
    let q = (v.clamp(0.0, 1.0) * 65535.0).round() as u32;
    let shift = 16 - bits.min(16);
    let step = 1u32 << shift;
    let snapped = (((q + step / 2) / step) * step).min(65535);
    snapped as f32 / 65535.0
}

/// Bit-packed plane with a v3-style split: zigzag delta across frequency, then
/// the low byte and the remaining high bits in separate planes before LZ4.
///
/// This is what a real v4 coder would do at reduced depth — v3 exactly, minus
/// the bits nobody can see. Reported alongside the unsplit form because the
/// split is worth a few percent on its own.
fn split_plane_bytes(vals: &[u16], bits: u32, n_bars: usize) -> u64 {
    let shift = 32 - bits;
    let mask: u32 = (1u32 << bits) - 1;
    let mut lo: Vec<u8> = Vec::with_capacity(vals.len());
    let mut hi: Vec<u16> = Vec::with_capacity(vals.len());
    for row in vals.chunks(n_bars.max(1)) {
        let mut prev = 0i32;
        for &v in row {
            let diff = ((v as i32 - prev) << shift) >> shift;
            prev = v as i32;
            let z = (((diff << 1) ^ (diff >> 31)) as u32) & mask;
            lo.push((z & 0xFF) as u8);
            hi.push((z >> 8) as u16);
        }
    }
    let hi_bits = bits.saturating_sub(8);
    let mut payload = lz4_flex::compress_prepend_size(&lo);
    if hi_bits > 0 {
        payload.extend_from_slice(&lz4_flex::compress_prepend_size(&pack_bits16(
            &hi, hi_bits,
        )));
    }
    payload.len() as u64
}

/// Does a cached value need 16 bits?
///
/// The bars map the full 80 dB onto the plot height, so one pixel is `80/H` dB
/// and quantisation is invisible once there are at least `H` levels. A u16
/// gives 65 536 of them for a plot that is at most a few thousand pixels tall.
/// This measures what the slack is actually worth, two ways:
///
/// * **v3-as-is** — drop precision but keep the u16 payload and the shipping
///   coder. Costs no format change at all; the question is whether LZ4 can
///   recover the zeroed low bits through the delta and byte-plane split.
/// * **packed** — genuinely `bits` wide on disk. The floor, but a new coder.
#[test]
#[ignore]
fn bit_depth_sweep() {
    let _ = std::fs::create_dir_all(OUT_DIR);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let out = PathBuf::from(OUT_DIR).join(format!("depth_{stamp}.txt"));
    let mut w = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    eprintln!("depth report -> {}", out.display());

    let s = crate::spectrum::load_spectrum_settings().expect("no spectrum.json");
    let p = Params {
        n_bars: s.bar_count,
        min_freq: s.min_freq,
        max_freq: s.max_freq,
        pre_fps: s.pre_fps,
        aslt_cfg: s.aslt_cfg.clone(),
        pad_factor: s.pad_factor,
        overlap: s.overlap,
        window_fn: s.window_fn.clone(),
        interp: s.interp_mode.clone(),
    };

    const DEPTHS: [u32; 9] = [8, 9, 10, 11, 12, 13, 14, 15, 16];
    // Plot heights the choice would have to cover.
    const HEIGHTS: [(u32, &str); 4] = [
        (1080, "1080p"),
        (1440, "1440p"),
        (2160, "4K"),
        (4320, "8K"),
    ];

    let _ = writeln!(
        w,
        "bit-depth sweep — mono plane only\nbars {} | {:.0}-{:.0} Hz | pre_fps {}\n",
        p.n_bars, p.min_freq, p.max_freq, p.pre_fps
    );
    let _ = writeln!(w, "step and worst-case error per depth, over the 80 dB axis:\n");
    let _ = writeln!(w, "bits | levels | step dB | max err dB | px err @1080p | @1440p | @4K | @8K");
    let _ = writeln!(w, "--- | --- | --- | --- | --- | --- | --- | ---");
    for b in DEPTHS {
        let levels = (1u32 << b) - 1;
        let step = 80.0 / levels as f64;
        let err = step / 2.0;
        let px: Vec<String> = HEIGHTS
            .iter()
            .map(|(h, _)| format!("{:.3}", err * *h as f64 / 80.0))
            .collect();
        let _ = writeln!(
            w,
            "{b} | {levels} | {step:.5} | {err:.5} | {} | {} | {} | {}",
            px[0], px[1], px[2], px[3]
        );
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(MUSIC_DIR)
        .expect("Test Music Files")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|f| f.is_file())
        .collect();
    files.sort();

    let maps = [Mapping::Superlet, Mapping::Cqt];
    // [mapping][depth] -> (v3 bytes, packed bytes) summed over tracks
    let mut totals: Vec<Vec<(u64, u64)>> =
        maps.iter().map(|_| vec![(0u64, 0u64); DEPTHS.len()]).collect();
    let mut totals_a: Vec<Vec<u64>> = maps.iter().map(|_| vec![0u64; DEPTHS.len()]).collect();
    let mut totals_s: Vec<Vec<u64>> = maps.iter().map(|_| vec![0u64; DEPTHS.len()]).collect();

    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let (l_sig, r_sig, sr) = match decode_stereo(f) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(w, "\n### {name}\n  SKIPPED: {e}");
                continue;
            }
        };
        let m_sig: Vec<f32> = l_sig
            .iter()
            .zip(&r_sig)
            .map(|(&a, &b)| (a + b) / 2.0)
            .collect();
        drop(l_sig);
        drop(r_sig);
        let _ = writeln!(w, "\n\n### {name}   ({sr} Hz, {:.1} s)", m_sig.len() as f64 / sr as f64);
        eprintln!("=== {name}");

        for (mi, &mapping) in maps.iter().enumerate() {
            let t = Instant::now();
            let (raw, db_ref) = run(&m_sig, sr, &p, mapping);
            if raw.is_empty() {
                continue;
            }
            let frames = raw.len();
            let bars = raw[0].len();
            eprintln!("    {} {frames}x{bars} in {:.0}s", mapping.name(), t.elapsed().as_secs_f64());

            // The shipping pipeline, exactly: magnitude -> dB -> [0,1].
            let vals: Vec<Vec<f32>> = raw
                .iter()
                .map(|row| row.iter().map(|&m| cache_v(m, db_ref)).collect())
                .collect();
            drop(raw);

            let _ = writeln!(
                w,
                "\n  --- {} --- {frames} frames x {bars} bars",
                mapping.name()
            );
            let _ = writeln!(
                w,
                "  bits | v3 unaligned | v3 aligned | packed | packed+split"
            );
            let _ = writeln!(w, "  --- | --- | --- | --- | ---");

            for (di, &b) in DEPTHS.iter().enumerate() {
                let qu: Vec<Vec<f32>> = vals
                    .iter()
                    .map(|row| row.iter().map(|&v| requantise_unaligned(v, b)).collect())
                    .collect();
                let v3u = v3_bytes(&qu, "depth_u");
                drop(qu);

                let qa: Vec<Vec<f32>> = vals
                    .iter()
                    .map(|row| row.iter().map(|&v| requantise_aligned(v, b)).collect())
                    .collect();
                let v3a = v3_bytes(&qa, "depth_a");
                drop(qa);

                let levels = ((1u32 << b) - 1) as f32;
                let flat: Vec<u16> = vals
                    .iter()
                    .flat_map(|row| {
                        row.iter()
                            .map(move |&v| (v.clamp(0.0, 1.0) * levels).round() as u16)
                    })
                    .collect();
                let pk = packed_plane_bytes(&flat, b, bars);
                let sp = split_plane_bytes(&flat, b, bars);
                drop(flat);

                totals[mi][di].0 += v3u;
                totals[mi][di].1 += pk;
                totals_a[mi][di] += v3a;
                totals_s[mi][di] += sp;
                let _ = writeln!(
                    w,
                    "  {b} | {:.2} | {:.2} | {:.2} | {:.2}",
                    mb(v3u),
                    mb(v3a),
                    mb(pk),
                    mb(sp)
                );
            }
            let _ = w.flush();
        }
    }

    let _ = writeln!(w, "\n\n## Totals over all tracks (MB)\n");
    for (mi, &mapping) in maps.iter().enumerate() {
        let base = totals[mi][DEPTHS.len() - 1];
        let _ = writeln!(w, "\n### {}\n", mapping.name());
        let base_a = totals_a[mi][DEPTHS.len() - 1];
        let _ = writeln!(
            w,
            "bits | max err dB | v3 unaligned | v3 aligned | saving | packed+split | saving | covers"
        );
        let _ = writeln!(w, "--- | --- | --- | --- | --- | --- | --- | ---");
        for (di, &b) in DEPTHS.iter().enumerate() {
            let (v3u, _pk) = totals[mi][di];
            let v3a = totals_a[mi][di];
            let sp = totals_s[mi][di];
            let levels = (1u32 << b) - 1;
            let err = 80.0 / levels as f64 / 2.0;
            let covers = HEIGHTS
                .iter()
                .filter(|(h, _)| levels >= *h)
                .map(|(_, n)| *n)
                .collect::<Vec<_>>()
                .join(" ");
            let _ = writeln!(
                w,
                "{b} | {err:.4} | {:.1} | {:.1} | {:+.1}% | {:.1} | {:+.1}% | {}",
                mb(v3u),
                mb(v3a),
                v3a as f64 / base_a as f64 * 100.0 - 100.0,
                mb(sp),
                sp as f64 / base_a as f64 * 100.0 - 100.0,
                if covers.is_empty() { "-" } else { &covers }
            );
        }
        let _ = writeln!(
            w,
            "\nsavings are against the shipping 16-bit v3 coder ({:.1} MB).",
            mb(base_a)
        );
        let _ = base;
    }
    let _ = w.flush();
    eprintln!("done -> {}", out.display());
}

// ---------------------------------------------------------------------------
// Two convolution sets are enough
// ---------------------------------------------------------------------------

/// Feasibility proof for deriving mono without a third analysis pass.
///
/// # The claim this corrects
///
/// The handoff said the residual format costs a third analysis run, "+53 %
/// analysis time", because you cannot compute an error against something you
/// never computed. That is true of the *implementation* the study used — three
/// independent `analyze` calls — and false as a statement about the transform.
///
/// The earlier finding that a single per-bar `cos phi` cannot reproduce a
/// superlet bar is also correct, and does **not** constrain this. That result is
/// about a scalar attached to the *final* bar value, after the geometric mean
/// has already destroyed the per-member phase. Here nothing is attached to the
/// final value: the mono bar is built from the same members, in the same order,
/// with the complex sum taken before any magnitude is.
///
/// A Morlet convolution is linear, so for member `j`
///
/// ```text
/// M_j = (L_j + R_j) / 2
/// ```
///
/// holds exactly on the complex responses. Magnitude, floor and the weighted
/// geometric mean then run on `M_j` exactly as they would on a mono signal
/// analysed from scratch. So L, R and M all come out of **two** sets of channel
/// convolutions, and the third pass is an artefact of how the study was written.
///
/// # What this is not
///
/// A proof, not a licence to refactor. It exercises the direct per-frame path
/// and the whole-signal path that `Superlet::responses` selects — which for
/// these kernel lengths and hop is the overlap-save FFT route. It does **not**
/// exercise the GPU pipeline, the shared-signal-transform route with a
/// populated cache, or anything in `analyze_routed`s bar planning. Production
/// would have to carry complex responses through all of those, and that is a
/// separate piece of work.
///
/// Tolerances are numerical, not bit-exact: the two routes accumulate in
/// different orders and the derived path adds a complex mean before taking a
/// magnitude, so agreement is asserted in relative terms.
#[cfg(test)]
mod two_pass_tests {
    use super::*;

    const SR: f32 = 48_000.0;

    /// The complex counterpart of `Morlet::response`, tap for tap.
    ///
    /// Deliberately a transcription of the production function rather than a
    /// fresh derivation, including its edge handling: taps outside the signal
    /// are dropped and the envelope is renormalised over the survivors. If this
    /// drifts from the original the proof is worthless, so
    /// `the_complex_copy_reproduces_the_production_magnitude` holds it to it.
    fn morlet_complex(w: &Morlet, signal: &[f32], centre: isize) -> (f64, f64) {
        let n = signal.len() as isize;
        let base = centre - w.half as isize;
        let end = base + w.re.len() as isize;
        let lo = base.max(0);
        let hi = end.min(n);
        if hi <= lo {
            return (0.0, 0.0);
        }
        let mut acc_re = 0.0f64;
        let mut acc_im = 0.0f64;
        if base >= 0 && end <= n {
            for (k, (&wr, &wi)) in w.re.iter().zip(w.im.iter()).enumerate() {
                let s = signal[(base + k as isize) as usize];
                acc_re += (s * wr) as f64;
                acc_im += (s * wi) as f64;
            }
            return (2.0 * acc_re, 2.0 * acc_im);
        }
        let mut env_sum = 0.0f64;
        for i in lo..hi {
            let k = (i - base) as usize;
            let s = signal[i as usize];
            acc_re += (s * w.re[k]) as f64;
            acc_im += (s * w.im[k]) as f64;
            env_sum += w.env[k] as f64;
        }
        if env_sum <= 0.0 {
            return (0.0, 0.0);
        }
        (2.0 * acc_re / env_sum, 2.0 * acc_im / env_sum)
    }

    /// One superlet bar built from the two channels, without ever forming a
    /// mono signal — the thing being proved.
    fn derived_response(sl: &Superlet, l: &[f32], r: &[f32], centre: isize) -> f32 {
        let mut acc = 0.0f32;
        for (w, weight) in sl.wavelets.iter().zip(sl.weights.iter()) {
            let (lre, lim) = morlet_complex(w, l, centre);
            let (rre, rim) = morlet_complex(w, r, centre);
            // The complex mean, before any magnitude is taken. This is the
            // whole trick, and the reason the geometric mean is no obstacle:
            // the members are combined individually.
            let mre = (lre + rre) * 0.5;
            let mim = (lim + rim) * 0.5;
            let mag = mre.hypot(mim) as f32;
            acc += weight * mag.max(MAG_FLOOR).ln();
        }
        (acc / sl.total_weight).exp()
    }

    fn cfg() -> AsltConfig {
        AsltConfig {
            q_ratio: 2.0,
            max_window_s: 0.25,
            n_wavelets: 9,
            spread: 0.5,
            scale: crate::spectrum::freq_scale::FreqScale::Log,
        }
    }

    /// The production superlet for a bar of a 1024-bar grid.
    fn superlet_at(freq: f32) -> Superlet {
        let c = cfg();
        let grid = grid_q(1024, 20.0, 24_000.0);
        Superlet::new(freq, effective_q_at(freq, grid, &c), &c, SR)
    }

    /// Two channels with independent content, plus a common component.
    fn stereo(n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut l = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        let mut seed = 0x2545F491_4F6CDD1Du64;
        for i in 0..n {
            let t = i as f32 / SR;
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let noise = ((seed >> 40) as f32 / 8388608.0) - 1.0;
            let common = (std::f32::consts::TAU * 440.0 * t).sin() * 0.3;
            l.push(common + (std::f32::consts::TAU * 997.0 * t).sin() * 0.25 + noise * 0.05);
            r.push(common + (std::f32::consts::TAU * 1310.0 * t).cos() * 0.25 + noise * 0.04);
        }
        (l, r)
    }

    fn mono_of(l: &[f32], r: &[f32]) -> Vec<f32> {
        l.iter().zip(r).map(|(&a, &b)| (a + b) / 2.0).collect()
    }

    fn rel(a: f32, b: f32) -> f32 {
        let d = (a - b).abs();
        let s = a.abs().max(b.abs()).max(1e-12);
        d / s
    }

    #[test]
    fn the_complex_copy_reproduces_the_production_magnitude() {
        let n = 8192;
        let (l, _) = stereo(n);
        for freq in [60.0f32, 440.0, 3000.0, 12_000.0] {
            let sl = superlet_at(freq);
            for w in &sl.wavelets {
                // Interior, and both edges, where the envelope is renormalised.
                for centre in [0isize, 1, (n / 3) as isize, (n / 2) as isize, n as isize - 1] {
                    let (re, im) = morlet_complex(w, &l, centre);
                    let mine = re.hypot(im) as f32;
                    let theirs = w.response(&l, centre);
                    assert!(
                        rel(mine, theirs) < 1e-5,
                        "freq {freq}, centre {centre}: copy {mine} vs production {theirs}"
                    );
                }
            }
        }
    }

    #[test]
    fn mono_is_derivable_from_two_convolution_sets() {
        let n = 16_384;
        let (l, r) = stereo(n);
        let m = mono_of(&l, &r);
        let mut worst = 0.0f32;
        for freq in [40.0f32, 120.0, 440.0, 1500.0, 6000.0, 15_000.0] {
            let sl = superlet_at(freq);
            // Interior frames, and deliberately both edges, where the envelope
            // renormalisation is in play and a naive derivation goes wrong.
            for centre in [
                0isize,
                7,
                (n / 4) as isize,
                (n / 2) as isize,
                (3 * n / 4) as isize,
                n as isize - 8,
                n as isize - 1,
            ] {
                let direct = sl.response(&m, centre);
                let derived = derived_response(&sl, &l, &r, centre);
                let e = rel(direct, derived);
                worst = worst.max(e);
                assert!(
                    e < 1e-4,
                    "freq {freq}, centre {centre}: analysed {direct}, derived {derived} \
                     (relative {e})"
                );
            }
        }
        // In dB, which is the domain that matters, that bound is well under a
        // thousandth of a decibel.
        let db = 20.0 * (1.0 + worst).log10();
        assert!(db < 0.001, "worst relative error {worst} is {db} dB");
    }

    #[test]
    fn the_derivation_survives_total_cancellation() {
        // R = -L, so mono is silence. This is the case a magnitude-only method
        // cannot represent at all -- both channels are loud and the sum is
        // nothing -- and the case the per-bar `cos phi` oracle had to clamp.
        let n = 8192;
        let (l, _) = stereo(n);
        let r: Vec<f32> = l.iter().map(|&x| -x).collect();
        let m = mono_of(&l, &r);
        assert!(m.iter().all(|&x| x.abs() < 1e-6), "setup: mono is not silent");

        for freq in [60.0f32, 440.0, 5000.0] {
            let sl = superlet_at(freq);
            for centre in [0isize, (n / 2) as isize, n as isize - 1] {
                let direct = sl.response(&m, centre);
                let derived = derived_response(&sl, &l, &r, centre);
                // Both should be at the floor, and neither should be a NaN or
                // an infinity that a logarithm turned into one.
                assert!(direct.is_finite() && derived.is_finite());
                assert!(
                    direct < 1e-6 && derived < 1e-6,
                    "freq {freq}, centre {centre}: cancellation gave {direct} / {derived}"
                );
            }
        }
    }

    #[test]
    fn the_derivation_matches_the_whole_signal_route_too() {
        // `Superlet::responses` is what `analyze` actually calls, and for these
        // kernels and this hop it takes the overlap-save FFT route rather than
        // the direct loop above. The derivation has to agree with that as well,
        // or it only holds for a path production does not use.
        let n = 32_768;
        let (l, r) = stereo(n);
        let m = mono_of(&l, &r);
        let hop = hop_for_fps(SR as u32, 180.0);
        let frames = frame_count(n, hop);
        assert!(frames > 8, "setup: too few frames");

        for freq in [80.0f32, 700.0, 4000.0] {
            let sl = superlet_at(freq);
            let via_route = sl.responses(&m, hop, frames);
            let mut worst = 0.0f32;
            let mut worst_at = 0usize;
            for (fi, &routed) in via_route.iter().enumerate().take(frames) {
                let derived = derived_response(&sl, &l, &r, (fi * hop) as isize);
                let e = rel(routed, derived);
                if e > worst {
                    worst = e;
                    worst_at = fi;
                }
            }
            assert!(
                worst < 1e-3,
                "freq {freq}: worst relative error {worst} at frame {worst_at} \
                 (route {}, derived {})",
                via_route[worst_at],
                derived_response(&sl, &l, &r, (worst_at * hop) as isize)
            );
        }
    }
}

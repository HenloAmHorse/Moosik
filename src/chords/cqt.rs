//! The picture the chord model was trained on: librosa's `hybrid_cqt` (36
//! bins an octave from F#0, 288 bins, hop 512 at 22050 Hz) after its
//! `estimate_tuning`, rebuilt here bin for bin.
//!
//! Bins whose filter is longer than 512 samples are a true constant-Q
//! response: one 65536-point FFT a frame and each bin's kernel spectrum.
//! Shorter ones are librosa's pseudo-CQT: a Hann-windowed 1024-point STFT
//! magnitude weighted by each kernel's magnitude spectrum.

use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};

pub const SR: u32 = 22_050;
pub const HOP: usize = 512;
pub const BINS: usize = 288;
const BPO: f64 = 36.0;
const NFFT: usize = 1 << 16;
const PSEUDO_NFFT: usize = 1024;

fn fmin() -> f64 {
    440.0 * 2f64.powf((18.0 - 69.0) / 12.0) // F#0
}

/// Periodic Hann, as scipy's `get_window('hann', n)`.
fn hann(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos()).collect()
}

fn frames(len: usize) -> usize {
    1 + len / HOP
}

/// How far the recording sits from A440, in fractions of a 1/36-octave bin
/// (librosa's `estimate_tuning` with 36 bins an octave): STFT peaks between
/// 150 Hz and 4 kHz, the louder half of them, and the most common offset
/// from the grid in steps of 0.01.
pub fn tuning(y: &[f32]) -> f64 {
    const N: usize = 2048;
    let win = hann(N);
    let fft = FftPlanner::<f64>::new().plan_fft_forward(N);
    let lo = (150.0 * N as f64 / SR as f64).ceil() as usize;
    let hi = (4000.0 * N as f64 / SR as f64).ceil() as usize; // bins below this are < 4 kHz
    let peaks: Vec<(f64, f64)> = (0..frames(y.len()))
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut buf: Vec<Complex<f64>> = (0..N)
                .map(|j| {
                    let s = (i * 512 + j) as isize - (N / 2) as isize;
                    let x = if s >= 0 && (s as usize) < y.len() { y[s as usize] as f64 } else { 0.0 };
                    Complex::new(x * win[j], 0.0)
                })
                .collect();
            fft.process(&mut buf);
            let s: Vec<f64> = buf[..N / 2 + 1].iter().map(|c| c.norm()).collect();
            let top = s.iter().cloned().fold(0.0, f64::max);
            let t: Vec<f64> = s.iter().map(|&v| if v > 0.1 * top { v } else { 0.0 }).collect();
            let n = s.len();
            let mut out = Vec::new();
            for k in lo.max(1)..hi.min(n) {
                let peak = if k + 1 < n { t[k] > t[k - 1] && t[k] >= t[k + 1] } else { t[k] > t[k - 1] };
                if !peak {
                    continue;
                }
                let (a, b, c) = (s[k - 1], s[k], *s.get(k + 1).unwrap_or(&0.0));
                let den = 2.0 * b - a - c;
                let shift = if k + 1 < n && den.abs() > 1e-12 { 0.5 * (c - a) / den } else { 0.0 };
                let grad = if k + 1 < n { (c - a) / 2.0 } else { b - a };
                out.push(((k as f64 + shift) * SR as f64 / N as f64, b + 0.5 * grad * shift));
            }
            out
        })
        .collect();
    let mut mags: Vec<f64> = peaks.iter().filter(|p| p.0 > 0.0).map(|p| p.1).collect();
    if mags.is_empty() {
        return 0.0;
    }
    mags.sort_by(|a, b| a.total_cmp(b));
    let m = mags.len();
    let median = if m % 2 == 1 { mags[m / 2] } else { (mags[m / 2 - 1] + mags[m / 2]) / 2.0 };
    let mut counts = [0u32; 100];
    for &(f, mag) in &peaks {
        if f <= 0.0 || mag < median {
            continue;
        }
        let mut r = (BPO * (f / 27.5).log2()).rem_euclid(1.0);
        if r >= 0.5 {
            r -= 1.0;
        }
        counts[(((r + 0.5) * 100.0).floor() as usize).min(99)] += 1;
    }
    let best = (0..100).max_by_key(|&i| (counts[i], std::cmp::Reverse(i))).unwrap();
    -0.5 + best as f64 * 0.01
}

enum Kernel {
    /// First FFT bin and the conjugate kernel spectrum from there, scaled.
    Full(usize, Vec<Complex<f64>>),
    /// Magnitude weights over the pseudo STFT's bins.
    Pseudo(Vec<f64>),
}

fn kernels(tuning: f64) -> Vec<Kernel> {
    let r = 2f64.powf(1.0 / BPO);
    let alpha = (r * r - 1.0) / (r * r + 1.0);
    let mut planner = FftPlanner::<f64>::new();
    let big = planner.plan_fft_forward(NFFT);
    let small = planner.plan_fft_forward(PSEUDO_NFFT);
    (0..BINS)
        .into_par_iter()
        .map(|k| {
            let f = fmin() * 2f64.powf(tuning / BPO) * 2f64.powf(k as f64 / BPO);
            let l = SR as f64 / (alpha * f);
            // librosa.filters.wavelet: Hann times a phasor over arange(-l//2, l//2), L1-normalised
            let (n0, n1) = ((-l / 2.0).floor() as i64, (l / 2.0).floor() as i64);
            let w = hann((n1 - n0) as usize);
            let mut k: Vec<Complex<f64>> = (n0..n1)
                .zip(&w)
                .map(|(n, w)| Complex::from_polar(*w, std::f64::consts::TAU * f * n as f64 / SR as f64))
                .collect();
            let norm: f64 = k.iter().map(|c| c.norm()).sum();
            k.iter_mut().for_each(|c| *c /= norm);
            if 2f64.powf(l.log2().ceil()) < 2.0 * HOP as f64 {
                let mut buf = vec![Complex::new(0.0, 0.0); PSEUDO_NFFT];
                for (b, c) in buf.iter_mut().zip(&k) {
                    *b = c * (l / PSEUDO_NFFT as f64);
                }
                small.process(&mut buf);
                return Kernel::Pseudo(buf[..PSEUDO_NFFT / 2 + 1].iter().map(|c| c.norm() / (PSEUDO_NFFT as f64).sqrt()).collect());
            }
            let mut buf = vec![Complex::new(0.0, 0.0); NFFT];
            for (n, c) in (n0..n1).zip(&k) {
                buf[n.rem_euclid(NFFT as i64) as usize] = *c;
            }
            big.process(&mut buf);
            let spec = &buf[..NFFT / 2 + 1];
            let top = spec.iter().map(|c| c.norm()).fold(0.0, f64::max);
            let keep: Vec<usize> = (0..spec.len()).filter(|&i| spec[i].norm() > 1e-4 * top).collect();
            let (a, b) = (keep[0], keep[keep.len() - 1] + 1);
            Kernel::Full(a, spec[a..b].iter().map(|c| c.conj() * (l.sqrt() / NFFT as f64)).collect())
        })
        .collect()
}

/// |CQT| of 22050 Hz mono audio: one row of `BINS` per hop, frames centred on
/// multiples of the hop.
pub fn cqt(y: &[f32], tuning: f64) -> Vec<[f32; BINS]> {
    let ker = kernels(tuning);
    let mut planner = FftPlanner::<f64>::new();
    let big = planner.plan_fft_forward(NFFT);
    let small = planner.plan_fft_forward(PSEUDO_NFFT);
    let win = hann(PSEUDO_NFFT);
    let at = |s: isize| if s >= 0 && (s as usize) < y.len() { y[s as usize] as f64 } else { 0.0 };
    // Only the low bins of the big FFT are ever read.
    let top = ker.iter().map(|k| if let Kernel::Full(a, s) = k { a + s.len() } else { 0 }).max().unwrap_or(0);
    let n = frames(y.len());
    // Two frames share one complex FFT: frame i as the real part, i + 1 as
    // the imaginary part, split apart afterwards by conjugate symmetry.
    (0..n.div_ceil(2))
        .into_par_iter()
        .map_init(
            || (vec![Complex::new(0.0, 0.0); NFFT], vec![Complex::new(0.0, 0.0); big.get_inplace_scratch_len()], vec![Complex::new(0.0, 0.0); top]),
            |(z, scratch, x), pair| {
                let (c0, c1) = ((2 * pair * HOP) as isize, ((2 * pair + 1) * HOP) as isize);
                // Each frame's centre sample goes to index 0, where the kernels are centred.
                for (j, v) in z.iter_mut().enumerate() {
                    let o = if j < NFFT / 2 { j as isize } else { j as isize - NFFT as isize };
                    *v = Complex::new(at(c0 + o), at(c1 + o));
                }
                big.process_with_scratch(z, scratch);
                let mut rows = [[0f32; BINS]; 2];
                for (f, row) in rows.iter_mut().enumerate() {
                    if 2 * pair + f >= n {
                        break;
                    }
                    let c = if f == 0 { c0 } else { c1 };
                    for (k, v) in x.iter_mut().enumerate() {
                        let (a, b) = (z[k], z[(NFFT - k) % NFFT].conj());
                        *v = if f == 0 { (a + b) * 0.5 } else { (a - b) * Complex::new(0.0, -0.5) };
                    }
                    let mut p: Vec<Complex<f64>> =
                        (0..PSEUDO_NFFT).map(|j| Complex::new(at(c - (PSEUDO_NFFT / 2) as isize + j as isize) * win[j], 0.0)).collect();
                    small.process(&mut p);
                    let pm: Vec<f64> = p[..PSEUDO_NFFT / 2 + 1].iter().map(|c| c.norm()).collect();
                    for (out, k) in row.iter_mut().zip(&ker) {
                        *out = match k {
                            Kernel::Full(a, spec) => x[*a..*a + spec.len()].iter().zip(spec).map(|(x, k)| x * k).sum::<Complex<f64>>().norm() as f32,
                            Kernel::Pseudo(w) => w.iter().zip(&pm).map(|(w, m)| w * m).sum::<f64>() as f32,
                        };
                    }
                }
                rows
            },
        )
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .take(n)
        .collect()
}

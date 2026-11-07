//! Isolated research. Compile with rustc; no application imports/dependencies.
//! A symmetric-Hann CQT definition and a three-rectangle sliding candidate.
use std::f64::consts::{PI, TAU};
use std::hint::black_box;
use std::time::Instant;

#[cfg(feature = "fft-screen")]
mod fft_lane;
#[cfg(feature = "fft-screen")]
mod grouped_fft;

#[derive(Clone, Copy, Debug, Default)]
struct C {
    re: f64,
    im: f64,
}
impl C {
    fn cis(a: f64) -> Self {
        Self {
            re: a.cos(),
            im: a.sin(),
        }
    }
    fn add(self, b: Self) -> Self {
        Self {
            re: self.re + b.re,
            im: self.im + b.im,
        }
    }
    fn sub(self, b: Self) -> Self {
        Self {
            re: self.re - b.re,
            im: self.im - b.im,
        }
    }
    fn scale(self, x: f64) -> Self {
        Self {
            re: self.re * x,
            im: self.im * x,
        }
    }
    fn mul(self, b: Self) -> Self {
        Self {
            re: self.re * b.re - self.im * b.im,
            im: self.re * b.im + self.im * b.re,
        }
    }
    fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }
}

#[derive(Clone)]
struct Kernel {
    sr: u32,
    f: f64,
    h: usize,
    gain: f64,
    coeffs: Vec<C>,
}
impl Kernel {
    fn new(sr: u32, f: f64, bpo: f64, cap: Option<f64>) -> Self {
        assert!(sr > 0 && f.is_finite() && f > 0.0 && f < sr as f64 / 2.0);
        assert!(bpo.is_finite() && bpo > 0.0);
        let q = 1.0 / (2.0_f64.powf(1.0 / bpo) - 1.0);
        let mut h = (q * sr as f64 / (2.0 * f)).ceil() as usize;
        if let Some(s) = cap {
            assert!(s.is_finite() && s > 0.0);
            h = h.min((s * sr as f64 / 2.0).floor() as usize);
        }
        assert!((2..=2_000_000).contains(&h), "research support bound");
        let weights: Vec<f64> = (0..=2 * h)
            .map(|i| 0.5 + 0.5 * (PI * (i as f64 - h as f64) / h as f64).cos())
            .collect();
        let gain = 2.0 / weights.iter().sum::<f64>();
        let coeffs = weights
            .into_iter()
            .enumerate()
            .map(|(i, w)| C::cis(-TAU * f * (i as f64 - h as f64) / sr as f64).scale(w * gain))
            .collect();
        Self {
            sr,
            f,
            h,
            gain,
            coeffs,
        }
    }
    fn direct(&self, x: &[f64], t: usize) -> C {
        self.coeffs
            .iter()
            .enumerate()
            .fold(C::default(), |s, (i, c)| {
                s.add(c.scale(sample(x, t as i64 + i as i64 - self.h as i64)))
            })
    }
}
fn sample(x: &[f64], n: i64) -> f64 {
    if n < 0 {
        0.0
    } else {
        x.get(n as usize).copied().unwrap_or(0.0)
    }
}

/// Independent reference: no precomputed coefficients or shared window builder.
/// Neumaier summation avoids making the optimized accumulator its own oracle.
#[cfg(test)]
fn reference(x: &[f64], t: usize, sr: u32, f: f64, h: usize) -> C {
    fn add(sum: &mut f64, correction: &mut f64, v: f64) {
        let next = *sum + v;
        *correction += if sum.abs() >= v.abs() {
            (*sum - next) + v
        } else {
            (v - next) + *sum
        };
        *sum = next;
    }
    let (mut r, mut rc, mut im, mut ic, mut total) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for j in -(h as i64)..=h as i64 {
        let w = (PI * j as f64 / (2.0 * h as f64)).cos().powi(2);
        total += w;
        let a = TAU * f * j as f64 / sr as f64;
        let v = sample(x, t as i64 + j) * w;
        add(&mut r, &mut rc, v * a.cos());
        add(&mut im, &mut ic, -v * a.sin());
    }
    C {
        re: (r + rc) * 2.0 / total,
        im: (im + ic) * 2.0 / total,
    }
}

struct Rect {
    sum: C,
    rotate: C,
    outgoing: C,
    incoming: C,
}
impl Rect {
    fn new(x: &[f64], t: usize, h: usize, w: f64) -> Self {
        let sum = (-(h as i64)..=h as i64).fold(C::default(), |s, j| {
            s.add(C::cis(-w * j as f64).scale(sample(x, t as i64 + j)))
        });
        Self {
            sum,
            rotate: C::cis(w),
            outgoing: C::cis(w * h as f64),
            incoming: C::cis(-w * (h + 1) as f64),
        }
    }
    fn advance(&mut self, old: f64, new: f64) {
        self.sum = self
            .sum
            .sub(self.outgoing.scale(old))
            .add(self.incoming.scale(new))
            .mul(self.rotate);
    }
}

/// Constant memory per bin, after one full initialization. Work per advance is
/// linear in elapsed samples rather than support. Rebase periodically to limit
/// recursive roundoff. No claim of bit identity with the direct f64 expression.
struct Sliding {
    h: usize,
    gain: f64,
    t: usize,
    since_reset: usize,
    reset: usize,
    omega: f64,
    rect: [Rect; 3],
}
impl Sliding {
    fn new(k: &Kernel, x: &[f64], t: usize, reset: usize) -> Self {
        assert!(reset > 0);
        let w = TAU * k.f / k.sr as f64;
        let rect = [
            Rect::new(x, t, k.h, w),
            Rect::new(x, t, k.h, w - PI / k.h as f64),
            Rect::new(x, t, k.h, w + PI / k.h as f64),
        ];
        Self {
            h: k.h,
            gain: k.gain,
            t,
            since_reset: 0,
            reset,
            omega: w,
            rect,
        }
    }
    fn value(&self) -> C {
        self.rect[0]
            .sum
            .scale(0.5)
            .add(self.rect[1].sum.scale(0.25))
            .add(self.rect[2].sum.scale(0.25))
            .scale(self.gain)
    }
    fn at(&mut self, x: &[f64], t: usize) -> C {
        assert!(t >= self.t, "a seek requires reset, not reverse recurrence");
        if self.since_reset + t - self.t >= self.reset {
            let w = self.omega;
            self.rect = [
                Rect::new(x, t, self.h, w),
                Rect::new(x, t, self.h, w - PI / self.h as f64),
                Rect::new(x, t, self.h, w + PI / self.h as f64),
            ];
            self.t = t;
            self.since_reset = 0;
        } else {
            while self.t < t {
                let old = sample(x, self.t as i64 - self.h as i64);
                let new = sample(x, self.t as i64 + self.h as i64 + 1);
                for r in &mut self.rect {
                    r.advance(old, new);
                }
                self.t += 1;
                self.since_reset += 1;
            }
        }
        self.value()
    }
}

fn centres(sr: u32, fps: u32, frames: usize) -> Vec<usize> {
    assert!(fps > 0);
    (0..frames)
        .map(|i| ((i as u64 * sr as u64) / fps as u64) as usize)
        .collect()
}
fn fixture(sr: u32, len: usize, kind: usize, seed: u64) -> Vec<f64> {
    let mut state = seed | 1;
    (0..len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state >> 11) as f64 / ((1u64 << 53) as f64) * 2.0 - 1.0;
            let t = i as f64 / sr as f64;
            match kind {
                0 => 0.0,
                1 => {
                    if i == len / 2 {
                        1.0
                    } else {
                        0.0
                    }
                }
                2 => {
                    0.55 * (TAU * 55.0 * t + 0.17).sin()
                        + 0.2 * (TAU * 440.0 * t + 0.9).cos()
                        + 0.1 * (TAU * 7000.0 * t).sin()
                }
                3 => noise,
                4 => 0.3 * (TAU * (30.0 * t + 2000.0 * t * t)).sin() + 0.1 * noise,
                _ => {
                    if i < len / 3 || i > len * 2 / 3 {
                        0.0
                    } else {
                        0.7 * (TAU * 440.0 * t).sin()
                    }
                }
            }
        })
        .collect()
}
fn code(c: C) -> u16 {
    // Research-only visualization convention, matching ASLT's documented scale.
    // Not a call to production cache::quantise; integration still needs that gate.
    let db = (20.0 * (c.norm() * 0.25).log10()).max(-80.0);
    (((db + 80.0) / 80.0).clamp(0.0, 1.0) * 4095.0).round() as u16
}

fn bench() {
    println!(
        "kind,sr,bpo,freq,half_samples,frames,rep,direct_ms,sliding_ms,max_complex_error,changed_research_codes"
    );
    for sr in [44100, 48000, 96000] {
        let x = fixture(sr, sr as usize * 2, 4, 0x154155);
        let ts = centres(sr, 180, 360);
        for bpo in [24.0, 96.0, 2048.0 / (1000.0_f64.log2())] {
            for f in [20.0, 55.0, 440.0, 4000.0, 16000.0] {
                let k = Kernel::new(sr, f, bpo, None);
                // Warm both methods; construction excluded for both. Sliding
                // state initialization IS included in every timed evaluation.
                black_box(k.direct(&x, ts[0]));
                for rep in 0..3 {
                    let run_direct = || {
                        let start = Instant::now();
                        let out: Vec<_> = ts.iter().map(|&t| k.direct(black_box(&x), t)).collect();
                        (start.elapsed().as_secs_f64() * 1000.0, out)
                    };
                    let run_sliding = || {
                        let start = Instant::now();
                        let mut p = Sliding::new(&k, &x, 0, sr as usize);
                        let out: Vec<_> = ts.iter().map(|&t| p.at(black_box(&x), t)).collect();
                        (start.elapsed().as_secs_f64() * 1000.0, out)
                    };
                    let ((d, a), (s, b)) = if rep % 2 == 0 {
                        (run_direct(), run_sliding())
                    } else {
                        let b = run_sliding();
                        let a = run_direct();
                        (a, b)
                    };
                    let err = a
                        .iter()
                        .zip(&b)
                        .map(|(a, b)| a.sub(*b).norm())
                        .fold(0.0_f64, f64::max);
                    let codes = a
                        .iter()
                        .zip(&b)
                        .filter(|(a, b)| code(**a) != code(**b))
                        .count();
                    println!(
                        "chirp_noise,{sr},{bpo:.8},{f},{},{},{rep},{d:.6},{s:.6},{err:.12e},{codes}",
                        k.h,
                        ts.len()
                    );
                }
            }
        }
    }
}
fn geometry() {
    println!("bpo,q,frequency,uncapped_support_s,lookahead_s");
    for b in [
        12.0,
        24.0,
        48.0,
        96.0,
        1024.0 / 1000.0_f64.log2(),
        2048.0 / 1000.0_f64.log2(),
    ] {
        let q = 1.0 / (2.0_f64.powf(1.0 / b) - 1.0);
        for f in [20.0, 55.0, 440.0, 1000.0, 10000.0] {
            println!("{b:.8},{q:.8},{f},{:.8},{:.8}", q / f, q / (2.0 * f));
        }
    }
}
fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("bench") => bench(),
        Some("geometry") => geometry(),
        #[cfg(feature = "fft-screen")]
        Some("fft") => fft_lane::bench(),
        #[cfg(feature = "fft-screen")]
        Some("grouped") => grouped_fft::bench(),
        Some("bank") => bank(),
        _ => panic!("usage: cqt_probe geometry|bench"),
    }
}

/// Complete bank screen without retaining all dense kernels. Each bin's direct
/// plan is built once, processed, then released; planning is timed separately.
/// Hybrid threshold is fixed before these runs, not fitted on their results.
fn bank() {
    println!(
        "sr,bpo,seconds,bins,frames,rep,plan_ms,direct_ms,hybrid_ms,max_complex_error,changed_research_codes"
    );
    for (sr, bpo, seconds) in [(48000, 24.0, 4), (48000, 48.0, 2), (96000, 24.0, 2)] {
        let x = fixture(sr, sr as usize * seconds, 4, 0x155f);
        let ts = centres(sr, 180, seconds * 180);
        let bins = (bpo * (16000.0_f64 / 20.0).log2()).floor() as usize + 1;
        for rep in 0..3 {
            let (mut plan_ms, mut d_ms, mut h_ms, mut err, mut codes) = (0.0, 0.0, 0.0, 0.0_f64, 0);
            for i in 0..bins {
                let f = 20.0 * 2.0_f64.powf(i as f64 / bpo);
                let start = Instant::now();
                let k = Kernel::new(sr, f, bpo, None);
                plan_ms += start.elapsed().as_secs_f64() * 1000.0;
                let direct = || {
                    let s = Instant::now();
                    let v: Vec<_> = ts.iter().map(|&t| k.direct(black_box(&x), t)).collect();
                    (s.elapsed().as_secs_f64() * 1000.0, v)
                };
                let hybrid = || {
                    let s = Instant::now();
                    // At most 1 kHz for the first conservative screen. A measured
                    // planner can replace this only with separate validation.
                    let v: Vec<_> = if f <= 1000.0 {
                        let mut p = Sliding::new(&k, &x, 0, sr as usize);
                        ts.iter().map(|&t| p.at(black_box(&x), t)).collect()
                    } else {
                        ts.iter().map(|&t| k.direct(black_box(&x), t)).collect()
                    };
                    (s.elapsed().as_secs_f64() * 1000.0, v)
                };
                let ((d, a), (h, b)) = if rep % 2 == 0 {
                    (direct(), hybrid())
                } else {
                    let b = hybrid();
                    let a = direct();
                    (a, b)
                };
                d_ms += d;
                h_ms += h;
                for (a, b) in a.iter().zip(&b) {
                    err = err.max(a.sub(*b).norm());
                    codes += usize::from(code(*a) != code(*b));
                }
            }
            println!(
                "{sr},{bpo},{seconds},{bins},{},{rep},{plan_ms:.6},{d_ms:.6},{h_ms:.6},{err:.12e},{codes}",
                ts.len()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn direct_matches_independent_reference() {
        for sr in [44100, 48000, 96000] {
            for kind in 0..6 {
                for f in [55.0, 440.0, 7000.0] {
                    let x = fixture(sr, 4096, kind, 123 + kind as u64);
                    let k = Kernel::new(sr, f, 24.0, None);
                    for t in [0, 1, 1951, 4095, 4800] {
                        let a = k.direct(&x, t);
                        let b = reference(&x, t, sr, f, k.h);
                        assert!(
                            a.sub(b).norm() < 1e-11,
                            "sr={sr} kind={kind} f={f} t={t} err={}",
                            a.sub(b).norm()
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn sliding_matches_reference_across_rebases_edges_and_noninteger_hops() {
        for sr in [44100, 48000, 96000] {
            for kind in 0..6 {
                for f in [55.0, 440.0, 7000.0] {
                    let x = fixture(sr, 8192, kind, 987 + kind as u64);
                    let k = Kernel::new(sr, f, 24.0, None);
                    let mut p = Sliding::new(&k, &x, 0, 2000);
                    for t in centres(sr, 180, 40) {
                        let a = p.at(&x, t);
                        let b = reference(&x, t, sr, f, k.h);
                        assert!(
                            a.sub(b).norm() < 2e-10,
                            "sr={sr} kind={kind} f={f} t={t} err={}",
                            a.sub(b).norm()
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn tone_gain_phase_and_impulse_have_independent_analytic_checks() {
        let sr = 48000;
        let f = 1000.0;
        let k = Kernel::new(sr, f, 48.0, None);
        let t = k.h * 2;
        let x: Vec<_> = (0..t + k.h + 1)
            .map(|n| 0.7 * (TAU * f * n as f64 / sr as f64 + 0.3).cos())
            .collect();
        let want = C::cis(TAU * f * t as f64 / sr as f64 + 0.3).scale(0.7);
        assert!(k.direct(&x, t).sub(want).norm() < 2e-6);
        let mut impulse = vec![0.0; t + k.h + 1];
        impulse[t] = 1.0;
        let got = k.direct(&impulse, t);
        assert!((got.re - 2.0 / k.h as f64).abs() < 1e-14);
        assert!(got.im.abs() < 1e-14);
        assert!(k.direct(&impulse, t + k.h + 1).norm() < 1e-14);
    }
    #[test]
    fn complex_stereo_mix_is_linear_before_magnitude() {
        let k = Kernel::new(48000, 440.0, 24.0, None);
        let l = fixture(48000, 4096, 2, 9);
        let r: Vec<_> = l.iter().map(|v| -v).collect();
        let mix: Vec<_> = l.iter().zip(&r).map(|(l, r)| (l + r) * 0.5).collect();
        let lc = k.direct(&l, 2000);
        let rc = k.direct(&r, 2000);
        assert!(lc.add(rc).scale(0.5).sub(k.direct(&mix, 2000)).norm() < 1e-14);
        assert!(lc.norm() > 0.01);
        assert!(((lc.norm() + rc.norm()) * 0.5 - k.direct(&mix, 2000).norm()).abs() > 0.01);
    }
    #[test]
    fn frame_centres_do_not_drift_and_cap_is_explicit() {
        assert_eq!(centres(44100, 180, 181)[180], 44100);
        assert_eq!(centres(48000, 180, 181)[180], 48000);
        let low = Kernel::new(48000, 20.0, 96.0, None);
        let high = Kernel::new(48000, 200.0, 96.0, None);
        assert!((low.h as f64 / high.h as f64 - 10.0).abs() < 0.01);
        let capped = Kernel::new(48000, 20.0, 96.0, Some(0.25));
        assert_eq!(capped.h, 6000);
        assert!(low.h > capped.h);
    }
    #[test]
    #[should_panic(expected = "a seek requires reset")]
    fn backwards_seek_cannot_reuse_old_state() {
        let k = Kernel::new(48000, 440.0, 24.0, None);
        let x = fixture(48000, 4096, 2, 1);
        let mut p = Sliding::new(&k, &x, 100, 2000);
        p.at(&x, 99);
    }
    #[test]
    fn sliding_survives_loud_to_silent_transition_and_longer_run() {
        let sr = 48000;
        let mut x = fixture(sr, sr as usize * 12, 3, 0x155);
        for v in &mut x[sr as usize * 4..] {
            *v = 0.0;
        }
        for f in [20.0, 440.0, 16000.0] {
            let k = Kernel::new(sr, f, 24.0, None);
            let mut p = Sliding::new(&k, &x, 0, sr as usize);
            for t in centres(sr, 180, 2160) {
                let a = p.at(&x, t);
                // Independent oracle at selected frames, especially tail and
                // just around a full reset. This is not every output sample.
                if t % (sr as usize) < 600 || t > sr as usize * 10 {
                    assert!(a.sub(reference(&x, t, sr, f, k.h)).norm() < 2e-10);
                }
                if t > sr as usize * 4 + k.h {
                    assert_eq!(code(a), 0);
                }
            }
        }
    }
    #[test]
    fn stationary_response_varies_with_frequency_offset_and_support() {
        // Spectral check does not use the recurrence or its decomposition.
        let sr = 48000;
        for f in [110.0, 880.0] {
            let k = Kernel::new(sr, f, 48.0, None);
            let t = k.h * 2;
            let mut amps = Vec::new();
            for off in [0.0, 1.0, 2.0] {
                let tone = f + off * sr as f64 / (2 * k.h) as f64;
                let x: Vec<_> = (0..t + k.h + 1)
                    .map(|n| (TAU * tone * n as f64 / sr as f64).cos())
                    .collect();
                amps.push(reference(&x, t, sr, f, k.h).norm());
            }
            assert!((amps[0] - 1.0).abs() < 2e-6);
            assert!((amps[1] - 0.5).abs() < 2e-6);
            assert!(amps[2] < 2e-6);
        }
    }
}

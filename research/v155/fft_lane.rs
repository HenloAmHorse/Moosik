//! Dense f64 overlap-save comparator, one bin at a time. No sparsification,
//! decimation or pruned inverse. Not the final shared-input-FFT bank design.
use super::*;
use rustfft::{FftPlanner, num_complex::Complex};

pub(super) fn convolution(k: &Kernel, x: &[f64], ts: &[usize]) -> Vec<C> {
    if ts.is_empty() {
        return Vec::new();
    }
    assert!(ts.windows(2).all(|v| v[0] <= v[1]));
    let n = (2 * k.coeffs.len()).next_power_of_two();
    let payload = n - 2 * k.h;
    let mut planner = FftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);
    let mut scratch = vec![
        Complex::default();
        fwd.get_inplace_scratch_len()
            .max(inv.get_inplace_scratch_len())
    ];
    let mut spectrum = vec![Complex::default(); n];
    for (i, c) in k.coeffs.iter().enumerate() {
        // Circular convolution with the reversed centred correlation kernel.
        let idx = (n + k.h - i) % n;
        spectrum[idx] = Complex::new(c.re, c.im);
    }
    fwd.process_with_scratch(&mut spectrum, &mut scratch);
    let mut buf = vec![Complex::default(); n];
    let mut out = vec![C::default(); ts.len()];
    let mut row = 0;
    while row < ts.len() {
        let start = ts[row];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = Complex::new(sample(x, start as i64 - k.h as i64 + i as i64), 0.0);
        }
        fwd.process_with_scratch(&mut buf, &mut scratch);
        for (v, h) in buf.iter_mut().zip(&spectrum) {
            *v *= h;
        }
        inv.process_with_scratch(&mut buf, &mut scratch);
        while row < ts.len() && ts[row] - start < payload {
            let v = buf[k.h + ts[row] - start];
            out[row] = C {
                re: v.re / n as f64,
                im: v.im / n as f64,
            };
            row += 1;
        }
    }
    out
}

pub(super) fn bench() {
    println!(
        "sr,bpo,freq,seconds,rep,direct_ms,fft_ms,max_complex_error,changed_research_codes,n,explicit_work_buffers_bytes"
    );
    for sr in [48000, 96000] {
        for bpo in [24.0, 96.0] {
            for f in [55.0, 440.0, 4000.0] {
                let x = fixture(sr, sr as usize * 4, 4, 0x155);
                let ts = centres(sr, 180, 720);
                let k = Kernel::new(sr, f, bpo, None);
                for rep in 0..3 {
                    let d = || {
                        let now = Instant::now();
                        let v: Vec<_> = ts.iter().map(|&t| k.direct(black_box(&x), t)).collect();
                        (now.elapsed().as_secs_f64() * 1000.0, v)
                    };
                    let fft = || {
                        let now = Instant::now();
                        let v = convolution(&k, black_box(&x), &ts);
                        (now.elapsed().as_secs_f64() * 1000.0, v)
                    };
                    let ((dt, a), (ft, b)) = if rep % 2 == 0 {
                        (d(), fft())
                    } else {
                        let b = fft();
                        let a = d();
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
                    let n = (2 * k.coeffs.len()).next_power_of_two();
                    // Spectrum + work + this rustfft plan's exact scratch length. This
                    // excludes planner allocations, input, coefficients and output.
                    let mut p = FftPlanner::<f64>::new();
                    let sc = p
                        .plan_fft_forward(n)
                        .get_inplace_scratch_len()
                        .max(p.plan_fft_inverse(n).get_inplace_scratch_len());
                    println!(
                        "{sr},{bpo},{f},4,{rep},{dt:.6},{ft:.6},{err:.12e},{codes},{n},{}",
                        (2 * n + sc) * 16
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_overlap_save_matches_independent_complex_reference() {
        for sr in [44100, 48000, 96000] {
            for f in [55.0, 440.0, 7000.0] {
                for kind in [1, 2, 3, 4] {
                    let x = fixture(sr, 8192, kind, 0x155 + kind as u64);
                    let k = Kernel::new(sr, f, 24.0, None);
                    let ts = centres(sr, 180, 50);
                    let ys = convolution(&k, &x, &ts);
                    for (&t, y) in ts.iter().zip(ys) {
                        assert!(y.sub(reference(&x, t, sr, f, k.h)).norm() < 2e-11);
                    }
                }
            }
        }
    }
}

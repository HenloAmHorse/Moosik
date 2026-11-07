//! Research: dense CQT bank sharing each input FFT among same-size kernels.
//! The byte limit bounds explicit FFT/kernel buffer payloads ONLY, not RSS,
//! planner internals, input audio, output matrices or allocator overhead.
use super::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::BTreeMap;

#[derive(Debug, Default)]
struct Stats {
    input_ffts: usize,
    kernel_ffts: usize,
    inverse_ffts: usize,
    batches: usize,
    peak_explicit_bytes: usize,
}

fn run(
    x: &[f64],
    freqs: &[f64],
    sr: u32,
    bpo: f64,
    ts: &[usize],
    limit: usize,
) -> Result<(Vec<Vec<C>>, Stats), &'static str> {
    assert!(ts.windows(2).all(|v| v[0] <= v[1]));
    assert!(bpo.is_finite() && bpo > 0.0);
    let q = 1.0 / (2.0_f64.powf(1.0 / bpo) - 1.0);
    let mut groups: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for (bi, &f) in freqs.iter().enumerate() {
        assert!(f.is_finite() && f > 0.0 && f < sr as f64 / 2.0);
        let h = (q * sr as f64 / (2.0 * f)).ceil() as usize;
        if !(2..=2_000_000).contains(&h) {
            return Err("support outside research bound");
        }
        let n = (2 * (2 * h + 1)).next_power_of_two();
        groups.entry(n).or_default().push((bi, h));
    }
    let mut out = vec![vec![C::default(); ts.len()]; freqs.len()];
    let mut stats = Stats::default();
    for (n, group) in groups {
        let max_h = group.iter().map(|(_, h)| *h).max().unwrap();
        let payload = n - 2 * max_h;
        // Planner lifetime is one size group. Its internal allocations are
        // deliberately not represented as a bound on total process memory.
        let mut planner = FftPlanner::<f64>::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        let sc = fwd
            .get_inplace_scratch_len()
            .max(inv.get_inplace_scratch_len());
        let common = (2 * n + sc) * 16;
        // Kernel::new transient weights + coefficients <=24*(2h+1) bytes;
        // n>=2*(2h+1), so one n*16 reserve is a conservative payload bound.
        let transient = n * 16;
        let Some(remaining) = limit.checked_sub(common + transient) else {
            return Err("FFT workspace budget too small");
        };
        let batch_len = (remaining / (n * 16)).min(group.len());
        if batch_len == 0 {
            return Err("FFT workspace budget too small");
        }
        let mut input = vec![Complex::default(); n];
        let mut product = vec![Complex::default(); n];
        let mut scratch = vec![Complex::default(); sc];
        for batch in group.chunks(batch_len) {
            stats.batches += 1;
            stats.peak_explicit_bytes = stats
                .peak_explicit_bytes
                .max(common + transient + batch.len() * n * 16);
            let mut kernels = Vec::with_capacity(batch.len());
            for &(bi, h) in batch {
                let k = Kernel::new(sr, freqs[bi], bpo, None);
                assert_eq!(k.h, h);
                let mut spectrum = vec![Complex::default(); n];
                for (i, c) in k.coeffs.iter().enumerate() {
                    spectrum[(n + h - i) % n] = Complex::new(c.re, c.im);
                }
                fwd.process_with_scratch(&mut spectrum, &mut scratch);
                stats.kernel_ffts += 1;
                kernels.push(spectrum);
            }
            let mut row = 0;
            while row < ts.len() {
                let first = row;
                let start = ts[row];
                for (i, v) in input.iter_mut().enumerate() {
                    *v = Complex::new(sample(x, start as i64 - max_h as i64 + i as i64), 0.0);
                }
                fwd.process_with_scratch(&mut input, &mut scratch);
                stats.input_ffts += 1;
                while row < ts.len() && ts[row] - start < payload {
                    row += 1;
                }
                for (&(bi, _), kernel) in batch.iter().zip(&kernels) {
                    for ((v, a), b) in product.iter_mut().zip(&input).zip(kernel) {
                        *v = *a * *b;
                    }
                    inv.process_with_scratch(&mut product, &mut scratch);
                    stats.inverse_ffts += 1;
                    for ri in first..row {
                        let v = product[max_h + ts[ri] - start];
                        out[bi][ri] = C {
                            re: v.re / n as f64,
                            im: v.im / n as f64,
                        };
                    }
                }
            }
        }
    }
    Ok((out, stats))
}

fn per_bin(x: &[f64], freqs: &[f64], sr: u32, bpo: f64, ts: &[usize]) -> Vec<Vec<C>> {
    freqs
        .iter()
        .map(|&f| {
            let k = Kernel::new(sr, f, bpo, None);
            super::fft_lane::convolution(&k, x, ts)
        })
        .collect()
}

pub(super) fn bench() {
    println!(
        "sr,bpo,bins,seconds,limit_bytes,rep,per_bin_ms,grouped_ms,max_complex_error,changed_research_codes,input_ffts,kernel_ffts,inverse_ffts,batches,peak_explicit_bytes"
    );
    for (sr, bpo) in [(48000, 24.0), (48000, 48.0), (96000, 24.0)] {
        let x = fixture(sr, sr as usize * 4, 4, 0x155c);
        let ts = centres(sr, 180, 720);
        let count = (bpo * (8000.0_f64 / 55.0).log2()).floor() as usize + 1;
        let freqs: Vec<_> = (0..count)
            .map(|i| 55.0 * 2.0_f64.powf(i as f64 / bpo))
            .collect();
        for limit in [32 * 1024 * 1024, 64 * 1024 * 1024] {
            for rep in 0..3 {
                let a = || {
                    let t = Instant::now();
                    let v = per_bin(black_box(&x), &freqs, sr, bpo, &ts);
                    (t.elapsed().as_secs_f64() * 1000.0, v)
                };
                let b = || {
                    let t = Instant::now();
                    let v = run(black_box(&x), &freqs, sr, bpo, &ts, limit).unwrap();
                    (t.elapsed().as_secs_f64() * 1000.0, v)
                };
                let ((at, aa), (bt, (bb, st))) = if rep % 2 == 0 {
                    (a(), b())
                } else {
                    let b = b();
                    let a = a();
                    (a, b)
                };
                let mut err = 0.0_f64;
                let mut codes = 0;
                for (a, b) in aa.iter().flatten().zip(bb.iter().flatten()) {
                    err = err.max(a.sub(*b).norm());
                    codes += usize::from(code(*a) != code(*b));
                }
                println!(
                    "{sr},{bpo},{count},4,{limit},{rep},{at:.6},{bt:.6},{err:.12e},{codes},{},{},{},{},{}",
                    st.input_ffts,
                    st.kernel_ffts,
                    st.inverse_ffts,
                    st.batches,
                    st.peak_explicit_bytes
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_input_dense_fft_matches_independent_direct_reference() {
        let freqs = [55.0, 57.0, 60.0, 440.0, 450.0, 4000.0];
        for sr in [44100, 48000, 96000] {
            for kind in [1, 3, 4] {
                let x = fixture(sr, 8192, kind, 17);
                let ts = centres(sr, 180, 50);
                let (out, stats) = run(&x, &freqs, sr, 24.0, &ts, 32 * 1024 * 1024).unwrap();
                assert_eq!(stats.kernel_ffts, freqs.len());
                assert!(stats.peak_explicit_bytes <= 32 * 1024 * 1024);
                for (bi, &f) in freqs.iter().enumerate() {
                    let k = Kernel::new(sr, f, 24.0, None);
                    for (&t, v) in ts.iter().zip(&out[bi]) {
                        assert!(v.sub(reference(&x, t, sr, f, k.h)).norm() < 2e-11);
                    }
                }
            }
        }
    }
    #[test]
    fn budget_changes_batching_not_the_result_and_small_budget_refuses() {
        let x = fixture(48000, 12000, 3, 57);
        let ts = centres(48000, 180, 40);
        let freqs: Vec<_> = (0..24).map(|i| 440.0 + i as f64).collect();
        let (a, sa) = run(&x, &freqs, 48000, 24.0, &ts, 768 * 1024).unwrap();
        let (b, sb) = run(&x, &freqs, 48000, 24.0, &ts, 8 * 1024 * 1024).unwrap();
        assert!(sa.batches > sb.batches);
        assert!(sa.input_ffts > sb.input_ffts);
        assert!(sa.peak_explicit_bytes <= 768 * 1024);
        for (a, b) in a.iter().flatten().zip(b.iter().flatten()) {
            assert_eq!(a.re.to_bits(), b.re.to_bits());
            assert_eq!(a.im.to_bits(), b.im.to_bits());
        }
        assert!(run(&x, &freqs, 48000, 24.0, &ts, 1024).is_err());
    }
}

use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn leaf(n: usize, direction: FftDirection) -> Box<dyn Fft<f64>> {
    match n {
        256 => Box::new(Butterfly256Avx64::<f64>::new(direction).unwrap()),
        512 => Box::new(Butterfly512Avx64::<f64>::new(direction).unwrap()),
        _ => unreachable!(),
    }
}

fn supported() -> bool {
    Butterfly256Avx64::<f64>::is_supported_by_cpu()
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn fixture(n: usize, kind: usize, seed: u64) -> Vec<Complex<f64>> {
    let mut seed = seed | 1;
    (0..n)
        .map(|i| {
            let a = next(&mut seed);
            let b = next(&mut seed);
            let r = (a >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0;
            let im = (b >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0;
            match kind {
                0 => Complex::new(
                    if i & 1 == 0 { 0.0 } else { -0.0 },
                    if i & 2 == 0 { -0.0 } else { 0.0 },
                ),
                1 => {
                    if i == n / 3 {
                        Complex::new(1.0, -0.5)
                    } else {
                        Complex::new(0.0, -0.0)
                    }
                }
                2 => Complex::new(r, im),
                3 => {
                    let angle = 2.0 * std::f64::consts::PI * 17.0 * i as f64 / n as f64;
                    Complex::new(angle.cos(), angle.sin())
                }
                4 => Complex::new(if i & 1 == 0 { 1e90 } else { -1e90 }, im * 1e-90),
                5 => Complex::new(
                    f64::from_bits((a & (1 << 63)) | (1 + (a & 0xffff))),
                    f64::from_bits((b & (1 << 63)) | (1 + (b & 0xffff))),
                ),
                6 => Complex::new(if i & 1 == 0 { 1.0 } else { -1.0 }, im * 1e-12),
                7 => Complex::new(r * 1e50, im * 1e-50),
                8 => Complex::new(r * f64::MIN_POSITIVE * 16.0, im * f64::MIN_POSITIVE * 16.0),
                9 => Complex::new(if i == n / 3 { f64::INFINITY } else { r }, im),
                10 => Complex::new(r, if i == n / 5 { f64::NAN } else { im }),
                _ => Complex::new(r * f64::MAX * 0.125, im * f64::MAX * 0.125),
            }
        })
        .collect()
}

fn component_bits_equal(actual: f64, expected: f64) -> bool {
    if expected.is_nan() {
        actual.is_nan()
    } else {
        actual.to_bits() == expected.to_bits()
    }
}

fn assert_complex(actual: Complex<f64>, expected: Complex<f64>, label: &str) {
    assert!(
        component_bits_equal(actual.re, expected.re)
            && component_bits_equal(actual.im, expected.im),
        "{label}: actual={actual:?} [{:016x},{:016x}] expected={expected:?} [{:016x},{:016x}]",
        actual.re.to_bits(),
        actual.im.to_bits(),
        expected.re.to_bits(),
        expected.im.to_bits()
    );
}

struct Replay {
    buffer: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    out: Vec<Complex<f64>>,
}

impl Replay {
    fn new(plan: &dyn Fft<f64>) -> Self {
        Self {
            buffer: vec![Complex::new(0.0, 0.0); plan.len()],
            // Unformed adjacent SIMD lanes are deliberately non-finite. They
            // must not affect the selected complex lane's result.
            scratch: vec![Complex::new(f64::NAN, -f64::INFINITY); plan.get_inplace_scratch_len()],
            out: Vec::with_capacity(plan.len()),
        }
    }

    fn full(plan: &dyn Fft<f64>, input: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let mut buffer = input.to_vec();
        let mut scratch = vec![Complex::new(0.0, 0.0); plan.get_inplace_scratch_len()];
        plan.process_with_scratch(&mut buffer, &mut scratch);
        buffer
    }

    fn check(
        &mut self,
        plan: &dyn Fft<f64>,
        input: &[Complex<f64>],
        expected: &[Complex<f64>],
        needed: &[usize],
    ) {
        self.buffer.copy_from_slice(input);
        self.scratch.fill(Complex::new(f64::NAN, -f64::INFINITY));
        self.out
            .resize(needed.len(), Complex::new(f64::NAN, f64::NAN));
        plan.process_pruned_with_scratch(
            &mut self.buffer,
            &mut self.scratch,
            needed,
            &mut self.out,
            crate::PrunedWork::none(),
        );
        for (&index, &actual) in needed.iter().zip(&self.out) {
            assert_complex(
                actual,
                expected[index],
                &format!("N={} requested={needed:?} output={index}", plan.len()),
            );
        }
    }
}

#[test]
fn every_single_output_matches_full_bits_for_both_directions_and_fixtures() {
    if !supported() {
        return;
    }
    for n in [256, 512] {
        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            let plan = leaf(n, direction);
            assert!(plan.prunes());
            assert_eq!(plan.pruned_work_len(4), 0);
            let mut replay = Replay::new(&*plan);
            for kind in 0..12 {
                let input = fixture(n, kind, 0x155 + kind as u64);
                let expected = Replay::full(&*plan, &input);
                for index in 0..n {
                    replay.check(&*plan, &input, &expected, &[index]);
                }
            }
        }
    }
}

#[test]
fn shared_rows_mixed_rows_seeded_sets_and_whole_fallback_match_full_bits() {
    if !supported() {
        return;
    }
    for n in [256, 512] {
        let rows = if n == 256 { 8 } else { 16 };
        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            let plan = leaf(n, direction);
            let mut replay = Replay::new(&*plan);
            for kind in [0, 2, 5, 6, 8] {
                let input = fixture(n, kind, 0xc0ffee + kind as u64);
                let expected = Replay::full(&*plan, &input);
                for index in 0..n - 4 {
                    replay.check(&*plan, &input, &expected, &[index, index + 1, index + 2]);
                    replay.check(
                        &*plan,
                        &input,
                        &expected,
                        &[index, index + 1, index + 2, index + 3],
                    );
                }
                for index in 0..n - rows * 3 {
                    replay.check(
                        &*plan,
                        &input,
                        &expected,
                        &[index, index + rows, index + 2 * rows, index + 3 * rows],
                    );
                }
                let mut seed = 0xbad5eed;
                for count in [2, 3, 4] {
                    for _ in 0..128 {
                        let mut set = std::collections::BTreeSet::new();
                        while set.len() < count {
                            set.insert(next(&mut seed) as usize % n);
                        }
                        let needed: Vec<_> = set.into_iter().collect();
                        replay.check(&*plan, &input, &expected, &needed);
                    }
                }
                for needed in [vec![], vec![0, 1, 2, 3, 4], (0..n).collect()] {
                    replay.check(&*plan, &input, &expected, &needed);
                }
            }
        }
    }
}

#[test]
#[ignore = "bounded exhaustive release gate: every sorted pair, both directions"]
fn every_sorted_pair_matches_full_bits() {
    if !supported() {
        return;
    }
    let mut calls = 0u64;
    for n in [256, 512] {
        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            let plan = leaf(n, direction);
            let input = fixture(n, 2, 0x1234567);
            let expected = Replay::full(&*plan, &input);
            let mut replay = Replay::new(&*plan);
            for a in 0..n {
                for b in a + 1..n {
                    replay.check(&*plan, &input, &expected, &[a, b]);
                    calls += 1;
                }
            }
        }
    }
    assert_eq!(calls, 326912);
    println!("f64_leaf_exhaustive_pairs,calls={calls},differences=0");
}

#[test]
fn product_entry_matches_old_mul_full_inverse_bits() {
    if !supported() {
        return;
    }
    for n in [256, 512] {
        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            let plan = leaf(n, direction);
            for kind in [0, 1, 2, 5, 6, 8] {
                let signal = fixture(n, kind, 0x155);
                let kernel = fixture(n, 2, 0x514);
                // This is CQT's retained baseline Mul, not an oracle formed
                // by the new product API's MulAssign implementation.
                let product: Vec<_> = signal.iter().zip(&kernel).map(|(s, h)| *s * *h).collect();
                let expected = Replay::full(&*plan, &product);
                let mut replay = Replay::new(&*plan);
                for index in 0..n {
                    let needed = [index];
                    replay.out.resize(1, Complex::new(0.0, 0.0));
                    plan.process_pruned_product_with_scratch(
                        &signal,
                        &kernel,
                        &mut replay.buffer,
                        &mut replay.scratch,
                        &needed,
                        &mut replay.out,
                        crate::PrunedWork::none(),
                    );
                    assert_complex(
                        replay.out[0],
                        expected[index],
                        "old Mul + whole inverse vs product pruning",
                    );
                    replay.buffer.copy_from_slice(&signal);
                    plan.process_pruned_inplace_product_with_scratch(
                        &kernel,
                        &mut replay.buffer,
                        &mut replay.scratch,
                        &needed,
                        &mut replay.out,
                        crate::PrunedWork::none(),
                    );
                    assert_complex(
                        replay.out[0],
                        expected[index],
                        "old Mul + whole inverse vs inplace product pruning",
                    );
                }
            }
        }
    }
}

#[test]
fn contract_failures_happen_before_any_write_and_128_stays_whole() {
    if !supported() {
        return;
    }
    let small = Butterfly128Avx64::<f64>::new(FftDirection::Inverse).unwrap();
    assert!(!small.prunes());
    for n in [256, 512] {
        let plan = leaf(n, FftDirection::Inverse);
        for case in 0..8 {
            let signal = fixture(n, 2, 0x123);
            let kernel = fixture(n, 2, 0x456);
            let mut buffer = signal.clone();
            let mut scratch = fixture(n, 2, 0x789);
            let mut out = vec![Complex::new(123.0, -456.0); 2];
            let before = (buffer.clone(), scratch.clone(), out.clone());
            let error = catch_unwind(AssertUnwindSafe(|| match case {
                0 => plan.process_pruned_with_scratch(
                    &mut buffer[..n - 1],
                    &mut scratch,
                    &[1, 2],
                    &mut out,
                    crate::PrunedWork::none(),
                ),
                1 => plan.process_pruned_with_scratch(
                    &mut buffer,
                    &mut scratch[..n - 1],
                    &[1, 2],
                    &mut out,
                    crate::PrunedWork::none(),
                ),
                2 => plan.process_pruned_with_scratch(
                    &mut buffer,
                    &mut scratch,
                    &[2, 2],
                    &mut out,
                    crate::PrunedWork::none(),
                ),
                3 => plan.process_pruned_with_scratch(
                    &mut buffer,
                    &mut scratch,
                    &[2, 1],
                    &mut out,
                    crate::PrunedWork::none(),
                ),
                4 => plan.process_pruned_with_scratch(
                    &mut buffer,
                    &mut scratch,
                    &[1, n],
                    &mut out,
                    crate::PrunedWork::none(),
                ),
                5 => plan.process_pruned_with_scratch(
                    &mut buffer,
                    &mut scratch,
                    &[1],
                    &mut out,
                    crate::PrunedWork::none(),
                ),
                6 => plan.process_pruned_product_with_scratch(
                    &signal[..n - 1],
                    &kernel,
                    &mut buffer,
                    &mut scratch,
                    &[1, 2],
                    &mut out,
                    crate::PrunedWork::none(),
                ),
                _ => plan.process_pruned_inplace_product_with_scratch(
                    &kernel[..n - 1],
                    &mut buffer,
                    &mut scratch,
                    &[1, 2],
                    &mut out,
                    crate::PrunedWork::none(),
                ),
            }));
            assert!(error.is_err(), "contract case {case} did not fail");
            for (a, b) in buffer.iter().zip(&before.0) {
                assert_complex(*a, *b, "buffer changed before rejection");
            }
            for (a, b) in scratch.iter().zip(&before.1) {
                assert_complex(*a, *b, "scratch changed before rejection");
            }
            for (a, b) in out.iter().zip(&before.2) {
                assert_complex(*a, *b, "output changed before rejection");
            }
        }
    }
}

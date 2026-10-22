//! Round 12-E: the requested-output Butterfly256/512 leaves in the vendored
//! rustfft (`vendor/rustfft-6.4.1/src/avx/avx32_butterflies.rs`), held to the
//! unchanged full leaf and gather (`process_with_scratch`, which the change
//! does not touch) — the oracle is never the candidate.
//!
//! Equality policy (`Fft::process_pruned_with_scratch`'s qualification):
//! every finite component, signed zero and infinity bit for bit, and NaN-ness;
//! NaN payload and sign are not compared (LLVM leaves them unspecified for
//! both paths). The application's magnitude, floor and quantiser treat every
//! NaN alike. Floating-point status flags are not compared either.

use rustfft::num_complex::Complex;

type C = Complex<f32>;

/// Whether this CPU has the AVX+FMA plans the leaves live in (elsewhere the
/// planner builds other plans and these gates have nothing to test).
fn pruning_cpu() -> bool {
    #[cfg(target_arch = "x86_64")]
    { is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

/// Bits of every component, every NaN as one canonical NaN (see above).
fn bits(v: &[C]) -> Vec<(u32, u32)> {
    let b = |x: f32| if x.is_nan() { 0x7fc0_0000 } else { x.to_bits() };
    v.iter().map(|c| (b(c.re), b(c.im))).collect()
}

/// Input families: random, signed zeros, subnormal, large, mixed
/// magnitudes, exceptional (NaN with a payload, ±∞).
fn inputs(n: usize) -> Vec<(&'static str, Vec<C>)> {
    let mut seed = 0x0123_4567_89AB_CDEFu64;
    let mut rnd = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; ((seed >> 40) as f32 / 8_388_608.0) - 1.0 };
    vec![
        ("random", (0..n).map(|_| C { re: rnd(), im: rnd() }).collect()),
        ("signed_zero", (0..n).map(|i| C { re: if i % 3 == 0 { -0.0 } else { 0.0 }, im: if i % 5 == 0 { -0.0 } else { 0.0 } }).collect()),
        ("tiny", (0..n).map(|_| C { re: rnd() * 1e-39, im: rnd() * 1e-41 }).collect()),
        ("large", (0..n).map(|_| C { re: rnd() * 1e36, im: rnd() * 1e36 }).collect()),
        ("mixed", (0..n).map(|i| { let a = rnd(); C { re: a * [1.0, 1e-30, 1e30, 3.0][i % 4], im: -a } }).collect()),
        ("exceptional", (0..n).map(|i| match i % 7 {
            0 => C { re: f32::NAN, im: 1.0 },
            1 => C { re: f32::INFINITY, im: -2.0 },
            2 => C { re: -f32::INFINITY, im: f32::from_bits(0x7fc0_1234) },
            _ => C { re: rnd(), im: rnd() },
        }).collect()),
    ]
}

/// Requested sets for a leaf of `n`: every singleton; boundary, same-row,
/// same-output and census-shaped pairs; sampled 2- to 9-sets (5+ take the
/// full leaf).
fn sets(n: usize) -> Vec<Vec<usize>> {
    let mut s: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let pairs: &[(usize, usize)] = if n == 256 {
        &[(0, 255), (0, 1), (0, 8), (3, 4), (7, 8), (5, 13), (5, 133), (100, 228), (31, 32), (127, 128), (250, 251), (6, 254)]
    } else {
        &[(0, 511), (0, 1), (0, 16), (15, 16), (187, 432), (198, 443), (3, 19), (100, 356), (255, 256), (63, 64), (500, 510)]
    };
    s.extend(pairs.iter().map(|&(a, b)| vec![a, b]));
    let mut seed = 0xDEAD_BEEF_u64 ^ n as u64;
    for k in [2usize, 2, 2, 3, 3, 4, 4, 4, 5, 6, 9] {
        for _ in 0..6 {
            let mut v: Vec<usize> = Vec::new();
            while v.len() < k {
                seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                let x = (seed % n as u64) as usize;
                if !v.contains(&x) { v.push(x); }
            }
            v.sort_unstable();
            s.push(v);
        }
    }
    s.push((0..4).collect());
    s.push((n - 4..n).collect());
    s.push(vec![0, n / 4, n / 2, 3 * n / 4]);
    s
}

fn leaf_gate(n: usize) {
    let mut planner = rustfft::FftPlanner::<f32>::new();
    for inverse in [true, false] {
        let fft = if inverse { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) };
        let scratch_len = fft.get_inplace_scratch_len();
        for (name, x) in inputs(n) {
            let mut full = x.clone();
            fft.process_with_scratch(&mut full, &mut vec![C { re: 0.0, im: 0.0 }; scratch_len]);
            // Dirty, reused buffers across every set.
            let mut scratch: Vec<C> = (0..scratch_len).map(|i| C { re: f32::NAN, im: i as f32 * 1e-40 }).collect();
            for set in sets(n) {
                let mut buf = x.clone();
                let mut out = vec![C { re: f32::from_bits(0x7f80_0001), im: -7.0 }; set.len()];
                fft.process_pruned_with_scratch(&mut buf, &mut scratch, &set, &mut out, rustfft::PrunedWork::none());
                let want: Vec<C> = set.iter().map(|&i| full[i]).collect();
                assert_eq!(bits(&out), bits(&want), "n={n} {name} inverse={inverse} set={set:?}");
            }
        }
    }
}

/// Butterfly256: every singleton and the sampled sets (supported and full-leaf
/// fallback), both directions, every input family, dirty reused buffers.
#[test]
fn the_requested_output_leaf_256_matches_the_full_leaf() {
    if !pruning_cpu() { return; }
    leaf_gate(256);
}

/// Butterfly512, as the 256 gate.
#[test]
fn the_requested_output_leaf_512_matches_the_full_leaf() {
    if !pruning_cpu() { return; }
    leaf_gate(512);
}

/// The empty request leaves `out` empty and touches nothing it returns.
#[test]
fn an_empty_request_is_accepted_by_both_leaves() {
    if !pruning_cpu() { return; }
    let mut planner = rustfft::FftPlanner::<f32>::new();
    for n in [256usize, 512] {
        let fft = planner.plan_fft_inverse(n);
        let mut buf = vec![C { re: 1.0, im: 0.0 }; n];
        let mut scratch = vec![C { re: 0.0, im: 0.0 }; fft.get_inplace_scratch_len()];
        fft.process_pruned_with_scratch(&mut buf, &mut scratch, &[], &mut [], rustfft::PrunedWork::none());
    }
}

/// The contract is checked before anything is written, for both leaves:
/// unsorted, duplicate and out-of-range indices, wrong buffer, scratch and
/// output lengths all panic with the original messages.
#[test]
fn both_leaves_keep_the_pruned_contract() {
    if !pruning_cpu() { return; }
    let mut planner = rustfft::FftPlanner::<f32>::new();
    for n in [256usize, 512] {
        let fft = planner.plan_fft_inverse(n);
        let sl = fft.get_inplace_scratch_len();
        let z = C { re: 0.0, im: 0.0 };
        let case = |len: usize, scratch: usize, needed: Vec<usize>, out: usize, msg: &str| {
            let fft = fft.clone();
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let mut buf = vec![C { re: 1.0, im: 0.0 }; len];
                let mut s = vec![z; scratch];
                let mut o = vec![z; out];
                fft.process_pruned_with_scratch(&mut buf, &mut s, &needed, &mut o, rustfft::PrunedWork::none());
            }));
            let e = r.expect_err(msg);
            let text = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            assert!(text.contains(msg), "n={n}: expected '{msg}', got '{text}'");
        };
        case(n, sl, vec![9, 3], 2, "strictly increasing");
        case(n, sl, vec![3, 3], 2, "strictly increasing");
        case(n, sl, vec![n], 1, "out of range");
        case(n - 1, sl, vec![1], 1, "buffer length");
        case(n, sl - 1, vec![1], 1, "scratch shorter");
        case(n, sl, vec![1, 2], 1, "one output slot");
    }
}

/// Whole pruned inverses at the analysis's sizes (2^13 … 2^16), with
/// production-shaped requests (hop-spaced outputs of one overlap-save block,
/// as `InvWork::outputs` asks) at the census hops, against the full inverse
/// and gather. The leaves run inside (the 2^13/2^14/2^16 plans end in 256,
/// 2^15 in 512, with one or two outputs per base call).
#[test]
fn whole_pruned_inverses_match_the_full_inverse() {
    if !pruning_cpu() { return; }
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let mut seed = 0x5151_5151_u64;
    let mut rnd = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; ((seed >> 40) as f32 / 8_388_608.0) - 1.0 };
    for (n, k, hop) in [(8_192usize, 2_100usize, 245usize), (16_384, 4_000, 490), (32_768, 9_000, 490), (65_536, 22_051, 490),
                        (16_384, 3_000, 245), (32_768, 12_000, 533), (65_536, 20_000, 533)] {
        let inv = planner.plan_fft_inverse(n);
        let x: Vec<C> = (0..n).map(|_| C { re: rnd(), im: rnd() }).collect();
        let half = (k - 1) / 2;
        let mut full = x.clone();
        inv.process_with_scratch(&mut full, &mut vec![C { re: 0.0, im: 0.0 }; inv.get_inplace_scratch_len()]);
        for lo_frame in [0usize, 1, 7] {
            let first = (lo_frame * hop + half).max(k - 1);
            let needed: Vec<usize> = (0..).map(|j| first + j * hop).take_while(|&i| i < n).collect();
            let want: Vec<C> = needed.iter().map(|&i| full[i]).collect();
            let u = inv.pruned_work_len(needed.len());
            let mut wi = vec![0usize; 2 * u];
            let mut wv = vec![C { re: f32::NAN, im: 0.0 }; u];
            let mut buf = x.clone();
            let mut scratch = vec![C { re: 3.0, im: f32::NAN }; inv.get_inplace_scratch_len()];
            let mut out = vec![C { re: 0.0, im: 0.0 }; needed.len()];
            inv.process_pruned_with_scratch(&mut buf, &mut scratch, &needed, &mut out, rustfft::PrunedWork { idx: &mut wi, val: &mut wv });
            assert_eq!(bits(&out), bits(&want), "n={n} hop={hop} lo={lo_frame}");
        }
    }
}

/// Diagnostic builds only (`--features leaf-diagnostics`), run on its own
/// (`--exact`): the counters are process-wide. Shows that the whole-inverse
/// shapes above reach both requested-output leaves and that 5+-output and
/// empty requests reach the full leaf.
#[cfg(feature = "leaf-diagnostics")]
#[test]
#[ignore]
fn diag_whole_inverses_reach_both_leaves_and_the_fallback() {
    if !pruning_cpu() { return; }
    use rustfft::moosik_diagnostics as d;
    d::reset();
    whole_pruned_inverses_match_the_full_inverse();
    let [l256, f256, l512, f512] = d::counts();
    assert!(l256 > 0 && l512 > 0, "a requested-output leaf never ran: {:?}", d::counts());
    assert_eq!((f256, f512), (0, 0), "production-shaped requests fell back: {:?}", d::counts());
    d::reset();
    leaf_gate(256);
    leaf_gate(512);
    let [l256, f256, l512, f512] = d::counts();
    assert!(l256 > 0 && f256 > 0 && l512 > 0 && f512 > 0, "leaf and fallback not both reached: {:?}", d::counts());
    println!("LEAFDIAG ok");
}


/// K0: the standalone f32 AVX+FMA 256 and 512 plans report `prunes()`, so
/// callers that consult it take their requested-output path; 128 does not
/// (it has none), and the f64 butterflies are unchanged (no pruned path).
/// Mixed-radix plans keep reporting true.
#[test]
fn leaf_capabilities_are_reported_accurately() {
    if !pruning_cpu() { return; }
    let mut p32 = rustfft::FftPlanner::<f32>::new();
    let mut p64 = rustfft::FftPlanner::<f64>::new();
    for (n, want) in [(128usize, false), (256, true), (512, true), (2048, true), (16_384, true), (65_536, true)] {
        assert_eq!(p32.plan_fft_inverse(n).prunes(), want, "f32 inverse {n}");
        assert_eq!(p32.plan_fft_forward(n).prunes(), want, "f32 forward {n}");
    }
    // 1.5.5 adds exact f64 requested leaves; f32 expectations above stay
    // unchanged. Both directions report capability, while 128 stays whole.
    for (n, want) in [(128usize, false), (256, true), (512, true)] {
        assert_eq!(p64.plan_fft_inverse(n).prunes(), want, "f64 inverse {n}");
        assert_eq!(p64.plan_fft_forward(n).prunes(), want, "f64 forward {n}");
    }
}

/// K0: `InvWork::outputs` — the analysis's caller, which takes the pruned
/// entry only when `prunes()` is true — on standalone 256 and 512 inverses,
/// both operand forms, 1-4 outputs (the leaf) and more (its full-leaf
/// fallback), against the product, full inverse and gather with production's
/// `2·scale` scaling. This thread's inverse tally shows the pruned entry ran.
#[test]
fn inv_work_takes_the_leaf_for_standalone_256_and_512() {
    if !pruning_cpu() { return; }
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let mut seed = 0x0BAD_CAFE_u64;
    let mut rnd = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; ((seed >> 40) as f32 / 8_388_608.0) - 1.0 };
    for n in [256usize, 512] {
        let inv = planner.plan_fft_inverse(n);
        let s: Vec<C> = (0..n).map(|_| C { re: rnd(), im: rnd() }).collect();
        let h: Vec<C> = (0..n).map(|_| C { re: rnd(), im: rnd() }).collect();
        let scale = 1.0 / n as f32;
        for (hop, half, lo, outs) in [(n / 3, 5usize, 0usize, 1usize), (n / 4, 7, 1, 2), (n / 5, 3, 0, 4), (n / 9, 2, 0, 7)] {
            let at = |j: usize| (lo + j) * hop + half;
            // Oracle: MulAssign product, full inverse, gather, 2·scale.
            let mut prod = s.clone();
            for (x, b) in prod.iter_mut().zip(&h) { *x *= *b; }
            inv.process_with_scratch(&mut prod, &mut vec![C { re: 0.0, im: 0.0 }; inv.get_inplace_scratch_len()]);
            let want: Vec<C> = (0..outs).map(|j| { let c = prod[at(j)]; C { re: 2.0 * scale * c.re, im: 2.0 * scale * c.im } }).collect();
            for inplace in [false, true] {
                let mut iw = super::InvWork::new(&*inv, outs);
                let mut buf = if inplace { s.clone() } else { vec![C { re: f32::NAN, im: 1.0 }; n] };
                let mut scratch = vec![C { re: 9.0, im: f32::NAN }; inv.get_inplace_scratch_len()];
                let mut out = vec![C { re: 0.0, im: 0.0 }; outs];
                let t0 = super::INV_TALLY.with(|c| c.get());
                let op = if inplace { super::Operand::InPlace(&h) } else { super::Operand::Product(&s, &h) };
                iw.outputs(&*inv, op, &mut buf, &mut scratch, lo, hop, half, 0, scale, &mut out);
                let t1 = super::INV_TALLY.with(|c| c.get());
                assert_eq!(bits(&out), bits(&want), "n={n} outs={outs} inplace={inplace}");
                assert_eq!((t1[0] - t0[0], t1[1] - t0[1]), (1, 0), "n={n}: InvWork did not take the pruned entry");
            }
        }
    }
}


/// Requested sets whose rows at the top level of an `n`-point plan with
/// `rows` rows are exactly `subset` (bit r = row r), 1-3 outputs per row.
fn sets_for_rows(n: usize, rows: usize, subset: u32, seed: &mut u64) -> Vec<usize> {
    let mut v = Vec::new();
    for r in 0..rows {
        if subset >> r & 1 == 0 { continue; }
        *seed ^= *seed << 13; *seed ^= *seed >> 7; *seed ^= *seed << 17;
        let per = 1 + (*seed % 3) as usize;
        for q in 0..per {
            *seed ^= *seed << 13; *seed ^= *seed >> 7; *seed ^= *seed << 17;
            let col = ((*seed >> 8) as usize + q * 97) % (n / rows);
            v.push(col * rows + r);
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// The row subsets to test at a level of `rows` rows: every nonempty subset
/// for 4 rows; for 8 rows every singleton, both parities, halves, the
/// observed 4-of-8 and 6-of-8 shapes, complements of singletons, and a
/// sample of the rest. (All rows live takes the original dense pass.)
fn row_subsets(rows: usize) -> Vec<u32> {
    if rows == 4 { return (1u32..16).collect(); }
    let mut s: Vec<u32> = (0..8).map(|r| 1u32 << r).collect();
    s.extend([0x55, 0xAA, 0x0F, 0xF0, 0x33, 0xCC, 0x3F, 0xFC, 0x7E, 0xFF]);
    s.extend((0..8).map(|r| 0xFF ^ (1u32 << r)));
    let mut seed = 0x1357u64;
    for _ in 0..24 {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        let m = (seed % 255 + 1) as u32;
        s.push(m);
    }
    s
}

/// Top-level rows (the outermost radix) of the planner's power-of-two plans,
/// per precision. R13 closeout: R13 used the f32 table for f64 too, but the
/// f64 AVX planner builds a 128/256/512 base under a chain of radix-8 steps
/// (`plan_mixed_radix_base`/`plan_mixed_radix` for `f64`), so every large f64
/// top level has 8 rows. In diagnostic builds each call below checks these
/// against the level that actually ran (`moosik_diagnostics::levels_take`).
const F32_TOP_ROWS: [(usize, usize); 4] = [(8_192, 4), (16_384, 8), (32_768, 8), (65_536, 4)];
const F64_TOP_ROWS: [(usize, usize); 4] = [(8_192, 8), (16_384, 8), (32_768, 8), (65_536, 8)];

/// Whether the live-row column pass applies to a top level (mirrors
/// `live_rows_ok`: at least 8192 points, at most half the rows live; the
/// remainder condition holds for every length here).
#[cfg(feature = "leaf-diagnostics")]
fn expect_live(n: usize, rows: usize, subset: u32) -> bool {
    n >= 8_192 && 2 * subset.count_ones() as usize <= rows
}

/// Every tested live-row subset of `fft`'s top level (`rows` rows), pruned
/// against the unchanged full transform and gather, on three input families
/// (random; signed zeros; random with ±0 and ±∞ entries); dirty scratch and
/// work. Returns (live, dense) top-level passes taken by *this* call, read
/// from this thread's level records in diagnostic builds (each call is also
/// checked: length, rows, precision, live mask, branch), else (0, 0).
fn live_column_run<T: rustfft::FftNum>(fft: &dyn rustfft::Fft<T>, rows: usize, conv: fn(f32) -> T) -> (usize, usize) {
    let n = fft.len();
    let mut seed = 0xACE1_u64 ^ n as u64 ^ (std::mem::size_of::<T>() as u64) << 32;
    let rnd = |s: &mut u64| { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; ((*s >> 40) as f32 / 8_388_608.0) - 1.0 };
    let random: Vec<Complex<T>> = (0..n).map(|_| Complex { re: conv(rnd(&mut seed)), im: conv(rnd(&mut seed)) }).collect();
    let zeros: Vec<Complex<T>> = (0..n).map(|i| Complex { re: conv(if i % 3 == 0 { -0.0 } else { 0.0 }), im: conv(if i % 5 == 0 { -0.0 } else { 0.0 }) }).collect();
    let mut special = random.clone();
    for (i, v) in special.iter_mut().enumerate() {
        if i % 7 == 0 { v.re = conv(-0.0); }
        if i % 11 == 0 { v.im = conv(0.0); }
    }
    special[n / 3].re = conv(f32::INFINITY);
    special[n / 2 + 1].im = conv(f32::NEG_INFINITY);
    #[allow(unused_mut)] // counted in diagnostic builds only
    let (mut live_n, mut dense_n) = (0usize, 0usize);
    #[cfg(feature = "leaf-diagnostics")]
    rustfft::moosik_diagnostics::levels_record(true);
    for (family, x) in [("random", random), ("signed_zero", zeros), ("special", special)] {
        let mut full = x.clone();
        fft.process_with_scratch(&mut full, &mut vec![Complex { re: conv(0.0), im: conv(0.0) }; fft.get_inplace_scratch_len()]);
        let mut scratch = vec![Complex { re: conv(f32::NAN), im: conv(3.0) }; fft.get_inplace_scratch_len()];
        for subset in row_subsets(rows) {
            let needed = sets_for_rows(n, rows, subset, &mut seed);
            let u = fft.pruned_work_len(needed.len());
            let mut wi = vec![7usize; 2 * u];
            let mut wv = vec![Complex { re: conv(f32::NAN), im: conv(0.0) }; u];
            let mut buf = x.clone();
            let mut out = vec![Complex { re: conv(f32::INFINITY), im: conv(0.0) }; needed.len()];
            #[cfg(feature = "leaf-diagnostics")]
            let _ = rustfft::moosik_diagnostics::levels_take();
            fft.process_pruned_with_scratch(&mut buf, &mut scratch, &needed, &mut out, rustfft::PrunedWork { idx: &mut wi, val: &mut wv });
            #[cfg(feature = "leaf-diagnostics")]
            {
                let top = rustfft::moosik_diagnostics::levels_take().first().copied();
                let want = rustfft::moosik_diagnostics::Level {
                    len: n, rows, live: subset, taken: expect_live(n, rows, subset), elem: std::mem::size_of::<T>(),
                };
                assert_eq!(top, Some(want), "top level as run, {family} subset {subset:#x}");
                if want.taken { live_n += 1 } else { dense_n += 1 }
            }
            for (k, &i) in needed.iter().enumerate() {
                let (a, b) = (out[k], full[i]);
                #[allow(clippy::eq_op)]
                let same = |p: T, q: T| (p != p && q != q) || format!("{p:?}") == format!("{q:?}");
                assert!(same(a.re, b.re) && same(a.im, b.im),
                        "n={n} rows={rows} {} {family} subset={subset:#x} index {i}: {a:?} vs {b:?}",
                        std::any::type_name::<T>());
            }
        }
    }
    #[cfg(feature = "leaf-diagnostics")]
    rustfft::moosik_diagnostics::levels_record(false);
    (live_n, dense_n)
}

/// R13 lane A: live-row column passes on the planner's own plans. Whole
/// pruned inverses and forwards at 2^13-2^16, every tested live-row subset of
/// each plan's real top level, f32 and f64 (each with its own geometry, see
/// `F32_TOP_ROWS`/`F64_TOP_ROWS`), against the full transform and gather.
/// Diagnostic builds also require, per direction, that this test's own calls
/// took both the live-row and the dense pass in each precision.
#[test]
fn live_row_column_passes_match_the_full_transform() {
    if !pruning_cpu() { return; }
    for inverse in [true, false] {
        let (mut p32, mut p64) = (rustfft::FftPlanner::<f32>::new(), rustfft::FftPlanner::<f64>::new());
        let (mut t32, mut t64) = ((0, 0), (0, 0));
        for (n, rows) in F32_TOP_ROWS {
            let fft = if inverse { p32.plan_fft_inverse(n) } else { p32.plan_fft_forward(n) };
            let (l, d) = live_column_run::<f32>(&*fft, rows, |v| v);
            t32 = (t32.0 + l, t32.1 + d);
        }
        for (n, rows) in F64_TOP_ROWS {
            let fft = if inverse { p64.plan_fft_inverse(n) } else { p64.plan_fft_forward(n) };
            let (l, d) = live_column_run::<f64>(&*fft, rows, |v| v as f64);
            t64 = (t64.0 + l, t64.1 + d);
        }
        #[cfg(feature = "leaf-diagnostics")]
        assert!(t32.0 > 0 && t32.1 > 0 && t64.0 > 0 && t64.1 > 0,
                "inverse={inverse}: live/dense top passes f32 {t32:?} f64 {t64:?}");
        let _ = (t32, t64);
    }
}

/// R13 closeout: a radix-4 f64 top level, which the f64 planner never builds
/// for these lengths — so it is built explicitly, with the planner's own
/// constructor, over a planner-built inner plan of a quarter the length
/// (diagnostic builds only: the constructor is exposed there). 16384 points
/// qualify for the live-row pass (with at most two of four rows live); 4096
/// points never do (the length gate), so every subset there takes the dense
/// pass. Both directions; exactness against the same plan's full transform;
/// this test's own level records show the branch on every call.
#[cfg(feature = "leaf-diagnostics")]
#[test]
fn f64_radix4_live_column_passes_match_the_full_transform() {
    if !pruning_cpu() { return; }
    use rustfft::moosik_diagnostics as d;
    let mut planner = rustfft::FftPlanner::<f64>::new();
    for inverse in [true, false] {
        for (n, want_live) in [(16_384usize, true), (4_096, false)] {
            let inner = if inverse { planner.plan_fft_inverse(n / 4) } else { planner.plan_fft_forward(n / 4) };
            let fft = d::mixed_radix4(inner).expect("AVX+FMA radix-4 f64 plan");
            assert_eq!(fft.len(), n);
            assert!(fft.prunes());
            assert_eq!(fft.fft_direction(), if inverse { rustfft::FftDirection::Inverse } else { rustfft::FftDirection::Forward });
            let (live, dense) = live_column_run::<f64>(&*fft, 4, |v| v as f64);
            if want_live {
                assert!(live > 0 && dense > 0, "n={n} inverse={inverse}: live {live} dense {dense}");
            } else {
                assert_eq!(live, 0, "n={n} inverse={inverse}: below the length gate");
                assert!(dense > 0);
            }
        }
    }
}

/// R13 lane A through the analysis's caller: `InvWork::outputs` on 2^14,
/// 2^15 and 2^16 inverses, product and in-place operand forms (the column
/// pass forms the products as it loads), with hop-shaped requests whose top
/// rows are sparse (even hops) and dense (odd hops), against the product,
/// full inverse and gather with production's `2·scale`.
#[test]
fn live_row_column_passes_through_inv_work() {
    if !pruning_cpu() { return; }
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let mut seed = 0xBEEF_u64;
    let mut rnd = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; ((seed >> 40) as f32 / 8_388_608.0) - 1.0 };
    for (n, k) in [(16_384usize, 4_000usize), (32_768, 9_000), (65_536, 22_051)] {
        let inv = planner.plan_fft_inverse(n);
        let s: Vec<C> = (0..n).map(|_| C { re: rnd(), im: rnd() }).collect();
        let h: Vec<C> = (0..n).map(|_| C { re: rnd(), im: rnd() }).collect();
        let mut prod = s.clone();
        for (x, b) in prod.iter_mut().zip(&h) { *x *= *b; }
        inv.process_with_scratch(&mut prod, &mut vec![C { re: 0.0, im: 0.0 }; inv.get_inplace_scratch_len()]);
        let scale = 1.0 / n as f32;
        let half = (k - 1) / 2;
        for hop in [490usize, 533, 245, 512] {
            let lo = 0usize;
            let base = half + lo * hop - (k - 1).min(half + lo * hop);
            let outs = (n - (k - 1)).div_ceil(hop).min(n / hop);
            let at = |j: usize| (lo + j) * hop + half - base;
            let outs = (0..outs).take_while(|&j| at(j) < n).count();
            let want: Vec<C> = (0..outs).map(|j| { let c = prod[at(j)]; C { re: 2.0 * scale * c.re, im: 2.0 * scale * c.im } }).collect();
            for inplace in [false, true] {
                let mut iw = super::InvWork::new(&*inv, outs);
                let mut buf = if inplace { s.clone() } else { vec![C { re: f32::NAN, im: 1.0 }; n] };
                let mut scratch = vec![C { re: 9.0, im: f32::NAN }; inv.get_inplace_scratch_len()];
                let mut out = vec![C { re: 0.0, im: 0.0 }; outs];
                let op = if inplace { super::Operand::InPlace(&h) } else { super::Operand::Product(&s, &h) };
                iw.outputs(&*inv, op, &mut buf, &mut scratch, lo, hop, half, base, scale, &mut out);
                assert_eq!(bits(&out), bits(&want), "n={n} hop={hop} inplace={inplace}");
            }
        }
    }
}


/// Diagnostic builds only, run alone: the lane-A gates above reach both the
/// live-row and the dense column pass.
#[cfg(feature = "leaf-diagnostics")]
#[test]
#[ignore]
fn diag_lane_a_gates_reach_live_and_dense_passes() {
    if !pruning_cpu() { return; }
    use rustfft::moosik_diagnostics as d;
    let _ = d::passes_take();
    live_row_column_passes_match_the_full_transform();
    live_row_column_passes_through_inv_work();
    let (live, dense) = d::passes_take();
    assert!(live > 0 && dense > 0, "live {live} dense {dense}");
    println!("LANEA_DIAG live={live} dense={dense}");
}

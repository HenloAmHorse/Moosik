//! The pruned inverse (measured in round 10 as R10-P3A with the fused
//! product; production since round 11) held to the full inverse, bit for bit,
//! at every consumed output — and the contract of its entry points.

use super::*;
use rustfft::num_complex::Complex;
use rustfft::{Fft, PrunedWork};


fn noise(n: usize, seed: &mut u64) -> Vec<Complex<f32>> {
    let mut next = || {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 40) as f32 / 8_388_608.0) - 1.0
    };
    (0..n).map(|_| Complex { re: next(), im: next() }).collect()
}

fn bits(v: &[Complex<f32>]) -> Vec<(u32, u32)> {
    v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

/// Caller-owned bookkeeping for a pruned call on `fft` with up to `needed`
/// outputs.
fn work_for(fft: &dyn Fft<f32>, needed: usize) -> (Vec<usize>, Vec<Complex<f32>>) {
    let u = fft.pruned_work_len(needed);
    (vec![0; 2 * u], vec![Complex { re: 0.0, im: 0.0 }; u])
}

fn pw<'a>(w: &'a mut (Vec<usize>, Vec<Complex<f32>>)) -> PrunedWork<'a, f32> {
    PrunedWork { idx: &mut w.0, val: &mut w.1 }
}

/// On this machine the analysis's plans prune: what the exactness tests below
/// exercise is the pruned code, not the full-transform default.
fn avx() -> bool {
    is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma")
}

/// Every power-of-two inverse length the analysis can plan (`fft_block_len`
/// of `FFT_MIN_KERNEL` up to `2^FFT_MAX_LOG2`), through the planner the
/// analysis uses — after other lengths have been planned, as in a run — and a
/// fresh one: all outputs; production-shaped selections (hop-spaced from an
/// offset, at the hops of 44.1, 48, 88.2 and 96 kHz at 180 fps); and random
/// subsets. Inputs: noise at several scales, a product of two spectra,
/// impulses, zeros and tiny values.
#[test]
fn the_pruned_inverse_is_the_full_inverse_at_every_wanted_output() {
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let lo = fft_block_len(FFT_MIN_KERNEL).trailing_zeros();
    let mut checked = 0usize;
    for fresh in [false, true] {
        let mut planner = rustfft::FftPlanner::<f32>::new();
        for lg in lo..=FFT_MAX_LOG2 {
            let n = 1usize << lg;
            let inv = if fresh {
                planner.plan_fft_inverse(n)
            } else {
                ASLT_PLANNER.with(|p| p.borrow_mut().plan_fft_inverse(n))
            };
            assert_eq!(inv.prunes(), avx(), "n 2^{lg}: the plan's pruning is not what this CPU should give");
            let fwd = planner.plan_fft_forward(n);
            let scratch_len = inv.get_inplace_scratch_len();
            let mut w = work_for(&*inv, n);
            let mut inputs: Vec<Vec<Complex<f32>>> = vec![noise(n, &mut seed)];
            let mut a = noise(n, &mut seed);
            let mut b = noise(n, &mut seed);
            fwd.process(&mut a);
            fwd.process(&mut b);
            inputs.push(a.iter().zip(&b).map(|(x, y)| { let mut z = *x; z *= *y; z }).collect());
            inputs.push(noise(n, &mut seed).iter().map(|c| Complex { re: c.re * 1e-30, im: c.im * 1e-30 }).collect());
            inputs.push(noise(n, &mut seed).iter().map(|c| Complex { re: c.re * 1e20, im: c.im * 1e20 }).collect());
            let mut imp = vec![Complex { re: 0.0f32, im: 0.0 }; n];
            imp[3] = Complex { re: 1.0, im: -2.0 };
            inputs.push(imp);
            inputs.push(vec![Complex { re: 0.0f32, im: 0.0 }; n]);

            let mut sets: Vec<Vec<usize>> = vec![(0..n).collect()];
            for hop in [245usize, 267, 490, 533] {
                for off in [0usize, 1, n / 3, n - 1 - (n - 1) / 2] {
                    let s: Vec<usize> = (0..).map(|t| off + t * hop).take_while(|&i| i < n).collect();
                    if !s.is_empty() { sets.push(s); }
                }
            }
            let mut r = (0..n).filter(|i| (i.wrapping_mul(2_654_435_761) >> 7) % 97 == 0).collect::<Vec<_>>();
            r.dedup();
            sets.push(r);
            sets.push(vec![n - 1]);

            for x in &inputs {
                let mut full = x.clone();
                let mut scratch = vec![Complex { re: 0.0f32, im: 0.0 }; scratch_len];
                inv.process_with_scratch(&mut full, &mut scratch);
                for s in &sets {
                    let mut buf = x.clone();
                    let mut scratch = vec![Complex { re: 7.0f32, im: -7.0 }; scratch_len];
                    let mut out = vec![Complex { re: f32::NAN, im: f32::NAN }; s.len()];
                    inv.process_pruned_with_scratch(&mut buf, &mut scratch, s, &mut out, pw(&mut w));
                    let want: Vec<Complex<f32>> = s.iter().map(|&i| full[i]).collect();
                    assert_eq!(bits(&out), bits(&want), "n 2^{lg} (fresh planner: {fresh}), {} outputs", s.len());
                    checked += s.len();
                }
            }
            // The fused entry point: product `a·b` formed in the first column
            // pass, against `x = a; x *= b` then the full inverse.
            for _ in 0..3 {
                let a = noise(n, &mut seed);
                let mut b = noise(n, &mut seed);
                fwd.process(&mut b);
                let mut full: Vec<Complex<f32>> = a.iter().zip(&b).map(|(x, y)| { let mut z = *x; z *= *y; z }).collect();
                let mut scratch = vec![Complex { re: 0.0f32, im: 0.0 }; scratch_len];
                inv.process_with_scratch(&mut full, &mut scratch);
                for s in &sets {
                    let mut buf = vec![Complex { re: f32::NAN, im: 1.0 }; n];
                    let mut out = vec![Complex { re: f32::NAN, im: f32::NAN }; s.len()];
                    inv.process_pruned_product_with_scratch(&a, &b, &mut buf, &mut scratch, s, &mut out, pw(&mut w));
                    let want: Vec<Complex<f32>> = s.iter().map(|&i| full[i]).collect();
                    assert_eq!(bits(&out), bits(&want), "fused, n 2^{lg} (fresh planner: {fresh}), {} outputs", s.len());
                    checked += s.len();
                }
                for s in &sets {
                    let mut buf = a.clone();
                    let mut out = vec![Complex { re: f32::NAN, im: f32::NAN }; s.len()];
                    inv.process_pruned_inplace_product_with_scratch(&b, &mut buf, &mut scratch, s, &mut out, pw(&mut w));
                    let want: Vec<Complex<f32>> = s.iter().map(|&i| full[i]).collect();
                    assert_eq!(bits(&out), bits(&want), "in-place fused, n 2^{lg} (fresh planner: {fresh}), {} outputs", s.len());
                    checked += s.len();
                }
            }
        }
    }
    println!("pruned inverse: {checked} outputs compared bit for bit");
}

/// The plans the pruned entry points can meet, by implementation: the AVX
/// mixed-radix family (prunes), `BluesteinsAvx` (its own non-pruning
/// overrides), and plans on the trait default (the scalar planner; the
/// scalar Bluestein). Bluestein under the AVX planner is recognised by its
/// scratch requirement, which only Bluestein's algorithm makes longer than
/// `2·len` among these plans.
fn contract_plans() -> Vec<(&'static str, std::sync::Arc<dyn Fft<f32>>)> {
    let mut plans: Vec<(&'static str, std::sync::Arc<dyn Fft<f32>>)> = Vec::new();
    let mut scalar = rustfft::FftPlannerScalar::<f32>::new();
    plans.push(("scalar 4096 (trait default)", scalar.plan_fft_inverse(4096)));
    let inner = scalar.plan_fft_forward(4096);
    plans.push(("scalar Bluestein 1031 (trait default)",
                std::sync::Arc::new(rustfft::algorithm::BluesteinsAlgorithm::new(1031, inner))));
    if avx() {
        let mut p = rustfft::FftPlannerAvx::<f32>::new().expect("avx planner");
        let mixed = p.plan_fft_inverse(8192);
        assert!(mixed.prunes(), "setup: the AVX 8192 plan does not prune");
        plans.push(("AVX mixed radix 8192", mixed));
        let blue = [1031usize, 2039, 4093, 8191, 16_381, 1019, 2027]
            .into_iter()
            .map(|n| p.plan_fft_inverse(n))
            .find(|f| f.get_inplace_scratch_len() > 2 * f.len())
            .expect("setup: no AVX Bluestein plan among the candidate primes");
        assert!(!blue.prunes(), "setup: Bluestein claims to prune");
        plans.push(("AVX Bluestein", blue));
    }
    plans
}

/// A valid call on every non-pruning plan: the full transform's values at the
/// requested indices, bit for bit, through all three entry points, with no
/// bookkeeping needed.
#[test]
fn every_fallback_plan_gives_the_full_transform_at_the_requested_outputs() {
    let mut seed = 0x51ED_27A3_9C0B_44F1u64;
    let mut fallbacks = 0;
    for (what, fft) in contract_plans() {
        let n = fft.len();
        if fft.prunes() { continue; }
        fallbacks += 1;
        assert_eq!(fft.pruned_work_len(n), 0, "{what}: a non-pruning plan asks for bookkeeping");
        let s = noise(n, &mut seed);
        let h = noise(n, &mut seed);
        let needed: Vec<usize> = (0..n).step_by(37).chain([n - 1]).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let mut prod: Vec<Complex<f32>> = s.iter().zip(&h).map(|(a, b)| { let mut z = *a; z *= *b; z }).collect();
        let mut scratch = vec![Complex { re: 0.0f32, im: 0.0 }; fft.get_inplace_scratch_len()];
        let mut full = s.clone();
        fft.process_with_scratch(&mut full, &mut scratch);
        fft.process_with_scratch(&mut prod, &mut scratch);
        let mut out = vec![Complex { re: f32::NAN, im: f32::NAN }; needed.len()];

        let mut buf = s.clone();
        fft.process_pruned_with_scratch(&mut buf, &mut scratch, &needed, &mut out, PrunedWork::none());
        assert_eq!(bits(&out), bits(&needed.iter().map(|&i| full[i]).collect::<Vec<_>>()), "{what}: pruned");
        let mut buf = vec![Complex { re: f32::NAN, im: 0.0 }; n];
        fft.process_pruned_product_with_scratch(&s, &h, &mut buf, &mut scratch, &needed, &mut out, PrunedWork::none());
        assert_eq!(bits(&out), bits(&needed.iter().map(|&i| prod[i]).collect::<Vec<_>>()), "{what}: product");
        let mut buf = s.clone();
        fft.process_pruned_inplace_product_with_scratch(&h, &mut buf, &mut scratch, &needed, &mut out, PrunedWork::none());
        assert_eq!(bits(&out), bits(&needed.iter().map(|&i| prod[i]).collect::<Vec<_>>()), "{what}: in-place product");
    }
    assert!(fallbacks >= 2, "setup: {fallbacks} fallback plans");
}

/// Every implementation refuses the same malformed calls, before touching
/// the buffer: short signal or kernel (the product fallbacks used to zip past
/// a short operand and leave the rest of the buffer unmultiplied), a
/// requested-index list whose length is not the output's, unsorted,
/// duplicated or out-of-range indices, a short buffer or scratch, and (on a
/// pruning plan) short bookkeeping.
#[test]
fn every_pruned_entry_point_enforces_one_contract() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let mut seed = 0x0DDB_A11C_0FFE_E123u64;
    let mut cases = 0;
    for (what, fft) in contract_plans() {
        let n = fft.len();
        let sl = fft.get_inplace_scratch_len();
        let good: Vec<usize> = vec![1, 5, n - 1];
        let s = noise(n, &mut seed);
        let h = noise(n, &mut seed);
        let mut wv = work_for(&*fft, n);
        let sized = |v: &[Complex<f32>], l: usize| { let mut x = v.to_vec(); x.resize(l, v[0]); x };
        // (name, buffer length, scratch length, needed, out length, s length, h length, work short)
        type Case = (&'static str, usize, usize, Vec<usize>, usize, usize, usize, bool);
        let mut bad: Vec<Case> = vec![
            ("short signal", n, sl, good.clone(), 3, n - 1, n, false),
            ("short kernel", n, sl, good.clone(), 3, n, n - 1, false),
            ("long kernel", n, sl, good.clone(), 3, n, n + 1, false),
            ("more indices than outputs", n, sl, good.clone(), 2, n, n, false),
            ("fewer indices than outputs", n, sl, good.clone(), 4, n, n, false),
            ("unsorted indices", n, sl, vec![5, 1, n - 1], 3, n, n, false),
            ("duplicate indices", n, sl, vec![1, 5, 5], 3, n, n, false),
            ("index out of range", n, sl, vec![1, 5, n], 3, n, n, false),
            ("short buffer", n - 1, sl, good.clone(), 3, n, n, false),
        ];
        if sl > 0 { bad.push(("short scratch", n, sl - 1, good.clone(), 3, n, n, false)); }
        if fft.pruned_work_len(3) > 0 { bad.push(("short bookkeeping", n, sl, good.clone(), 3, n, n, true)); }
        for (case, bl, scl, needed, ol, s_len, h_len, short_work) in bad {
            // Which entry points the case applies to: the operand lengths only
            // exist on the product forms.
            let entries: &[u8] = if s_len != n { &[1] } else if h_len != n { &[1, 2] } else { &[0, 1, 2] };
            for &e in entries {
                let fill = Complex { re: 3.5f32, im: -1.25 };
                let mut buf = vec![fill; bl];
                let mut scratch = vec![Complex { re: 0.0f32, im: 0.0 }; scl];
                let mut out = vec![Complex { re: 0.0f32, im: 0.0 }; ol];
                let (mut wi, mut wvv) = (wv.0.clone(), wv.1.clone());
                if short_work { wi.truncate(wi.len().saturating_sub(1)); wvv.clear(); }
                let r = catch_unwind(AssertUnwindSafe(|| {
                    let work = PrunedWork { idx: &mut wi, val: &mut wvv };
                    match e {
                        0 => fft.process_pruned_with_scratch(&mut buf, &mut scratch, &needed, &mut out, work),
                        1 => fft.process_pruned_product_with_scratch(&sized(&s, s_len), &sized(&h, h_len),
                                                                     &mut buf, &mut scratch, &needed, &mut out, work),
                        _ => fft.process_pruned_inplace_product_with_scratch(&sized(&h, h_len),
                                                                             &mut buf, &mut scratch, &needed, &mut out, work),
                    }
                }));
                let msg = match &r {
                    Err(p) => p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default(),
                    Ok(()) => panic!("{what}: {case} accepted by entry point {e}"),
                };
                assert!(msg.contains("pruned transform:") || msg.contains("pruned product:"), "{what}: {case} (entry {e}) refused by something other than the contract: {msg}");
                assert!(buf.iter().all(|c| c.re == fill.re && c.im == fill.im), "{what}: {case} (entry {e}) touched the buffer before refusing");
                cases += 1;
            }
        }
        // The valid call, through every entry point, after the refusals.
        let mut out = vec![Complex { re: 0.0f32, im: 0.0 }; 3];
        let mut scratch = vec![Complex { re: 0.0f32, im: 0.0 }; sl];
        let mut buf = s.clone();
        fft.process_pruned_with_scratch(&mut buf, &mut scratch, &good, &mut out, pw(&mut wv));
        fft.process_pruned_product_with_scratch(&s, &h, &mut buf, &mut scratch, &good, &mut out, pw(&mut wv));
        let mut buf = s.clone();
        fft.process_pruned_inplace_product_with_scratch(&h, &mut buf, &mut scratch, &good, &mut out, pw(&mut wv));
    }
    println!("{cases} malformed calls refused");
}

fn tally_now() -> [u64; 2] { INV_TALLY.with(|c| c.get()) }

/// `InvWork` is the pruned inverse on a plan that prunes and the original
/// code — product pass, whole transform, gather — on one that does not
/// (another CPU's plan), with the same values either way: each against that
/// plan's own full inverse, through both operand forms, and the tally says
/// which path ran.
#[test]
fn the_inverse_workspace_falls_back_to_the_whole_transform_on_a_plan_that_does_not_prune() {
    let mut seed = 0x7F4A_7C15_9E37_79B9u64;
    let n = 1usize << 13;
    let k = n / 2 - 101;
    let (hop, half) = (490usize, (k - 1) / 2);
    let cap = InvWork::cap(n, k, hop);
    let lo = (k - 1 - half).div_ceil(hop);
    let count = (0..).take_while(|j| (lo + j) * hop + half < n).count();
    assert!(count >= 2 && count <= cap, "setup: {count} outputs, cap {cap}");
    let scale = 1.0 / n as f32;
    let aslt_inv = ASLT_PLANNER.with(|p| p.borrow_mut().plan_fft_inverse(n));
    let scalar_inv = rustfft::FftPlannerScalar::<f32>::new().plan_fft_inverse(n);
    for (what, inv, prunes) in [("analysis planner", aslt_inv, avx()), ("scalar planner", scalar_inv, false)] {
        assert_eq!(inv.prunes(), prunes, "setup: {what}");
        let s = noise(n, &mut seed);
        let h = noise(n, &mut seed);
        let mut full: Vec<C32> = s.iter().zip(&h).map(|(a, b)| { let mut z = *a; z *= *b; z }).collect();
        let mut scratch = vec![C32 { re: 0.0, im: 0.0 }; inv.get_inplace_scratch_len()];
        inv.process_with_scratch(&mut full, &mut scratch);
        let want: Vec<C32> = (0..count).map(|j| {
            let c = full[(lo + j) * hop + half];
            C32 { re: 2.0 * scale * c.re, im: 2.0 * scale * c.im }
        }).collect();
        let mut iw = InvWork::new(&*inv, cap);
        for in_place in [false, true] {
            let mut buf = if in_place { s.clone() } else { vec![C32 { re: f32::NAN, im: 0.0 }; n] };
            let mut out = vec![C32 { re: f32::NAN, im: f32::NAN }; count];
            let t0 = tally_now();
            let op = if in_place { Operand::InPlace(&h) } else { Operand::Product(&s, &h) };
            iw.outputs(&*inv, op, &mut buf, &mut scratch, lo, hop, half, 0, scale, &mut out);
            let t1 = tally_now();
            assert_eq!(bits(&out), bits(&want), "{what}, in place {in_place}");
            assert_eq!([t1[0] - t0[0], t1[1] - t0[1]], if prunes { [1, 0] } else { [0, 1] },
                       "{what}, in place {in_place}: the wrong path ran");
        }
    }
}

/// A shared subbatch's worker slot holds everything a member's windows use:
/// once initialised, running members over further windows allocates nothing
/// — the pruned inverse's indices and bookkeeping, the gathered columns and
/// the fold included. Witnessed: the inverses ran pruned.
#[test]
fn a_shared_subbatch_slot_does_not_allocate_per_window() {
    let mut seed = 0x3C6E_F372_FE94_F82Bu64;
    let n = 1usize << 13;
    let k = n / 2 - 101;
    let (hop, half) = (490usize, (k - 1) / 2);
    let step = n - k + 1;
    let (fwd, inv) = ASLT_PLANNER.with(|p| {
        let mut p = p.borrow_mut();
        (p.plan_fft_forward(n), p.plan_fft_inverse(n))
    });
    let scratch_len = fwd.get_inplace_scratch_len().max(inv.get_inplace_scratch_len());
    let cap = InvWork::cap(n, k, hop);
    let frames = 400usize;
    // Joint: two channels; a window of two blocks.
    let spectra: Vec<(Vec<C32>, Vec<C32>)> = (0..4).map(|_| {
        let mut x = noise(n, &mut seed);
        fwd.process(&mut x);
        (x, Vec::new())
    }).collect();
    let mut h = noise(n, &mut seed);
    fwd.process(&mut h);
    let blocks = |j: usize| -> Vec<(usize, usize, usize)> {
        (j..j + 2).map(|b| {
            let base = b * step;
            let lo = (base + k - 1 - half).div_ceil(hop);
            let hi = (base + n - 1 - half) / hop;
            (base, lo, hi.min(frames - 1))
        }).collect()
    };
    let mut acc = Cols { mix: vec![0.0; frames], left: vec![0.0; frames], right: vec![0.0; frames] };
    let mut slot = SubbatchSlot::new(n, scratch_len, cap, 2, &*inv);
    let resident = slot.resident();
    assert!(resident <= SubbatchSlot::bytes(n, scratch_len, cap, 2, &*inv), "the slot holds more than it charges");
    let geom = (hop, half, 1.0 / n as f32);
    slot.member(&mut acc, &h, &blocks(0), &spectra, 2, &*inv, geom);
    let wins: Vec<Vec<(usize, usize, usize)>> = (1..6).map(blocks).collect();
    let t0 = tally_now();
    let allocs = {
        let before = crate::alloc_probe::arm();
        for win in &wins {
            slot.member(&mut acc, &h, win, &spectra, 2, &*inv, geom);
        }
        crate::alloc_probe::disarm(before)
    };
    let t1 = tally_now();
    assert_eq!(allocs, 0, "a slot allocated after its first window");
    assert_eq!(slot.resident(), resident, "a slot's buffers grew");
    assert_eq!((t1[0] - t0[0] > 0, t1[1] - t0[1]), (avx(), if avx() { 0 } else { 20 }),
               "the slot's inverses did not take the expected path");
}

/// The analysis is the analysis with the original full inverse, every f32 of
/// Mix, Left and Right, through the production selection — shared pass
/// (block budget 0), shared-block route and own route — and the work counts
/// witness which inverse each run took.
#[test]
fn analysis_with_the_pruned_inverse_is_bit_identical() {
    let sr = 44_100u32;
    let n = (sr as f32 * 3.0) as usize;
    let tau = std::f32::consts::TAU;
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut nz = || {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        ((seed >> 40) as f32 / 8_388_608.0) - 1.0
    };
    let l: Vec<f32> = (0..n).map(|i| 0.5 * (tau * 220.0 * i as f32 / sr as f32).sin() + 0.1 * nz()).collect();
    let r: Vec<f32> = (0..n).map(|i| -0.5 * (tau * 220.0 * i as f32 / sr as f32).sin() + 1e-6 * nz()).collect();
    let cfg = AsltConfig { q_ratio: 2.0, max_window_s: 0.25, n_wavelets: 9, spread: 0.5,
                           scale: super::super::freq_scale::FreqScale::Blend(0.5) };
    let hop = hop_for_fps(sr, 180.0);
    fn never(_: usize) -> bool { false }
    let run = |budget: usize, full_inverse: bool| {
        analyze_routed(Input::Joint { left: &l, right: &r }, sr, 256, 20.0, 22_050.0, hop, &cfg,
                       &|| true, &|_| {}, never, Seams { shared_budget: budget, full_inverse, ..Seams::production() })
    };
    let b = |p: &[Vec<f32>]| p.iter().map(|r| r.iter().map(|v| v.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>();
    for budget in [0usize, usize::MAX / 4] {
        let want = run(budget, true);
        assert!(!want.mix.is_empty());
        assert!(want.work.inverses_full > 0 && want.work.inverses_pruned == 0,
                "the reference did not take the whole inverse: {:?}", want.work);
        let got = run(budget, false);
        if avx() {
            assert!(got.work.inverses_pruned > 0 && got.work.inverses_full == 0,
                    "production did not take the pruned inverse: {:?}", got.work);
        }
        assert_eq!(got.work.inverses_pruned + got.work.inverses_full, want.work.inverses_full,
                   "the two runs took different numbers of inverses");
        assert_eq!(b(&got.mix), b(&want.mix), "mix, budget {budget}");
        assert_eq!(b(&got.left), b(&want.left), "left, budget {budget}");
        assert_eq!(b(&got.right), b(&want.right), "right, budget {budget}");
        println!("budget {budget}: shared members {}, size-seq groups {}, inverses {}",
                 got.work.shared_members, got.work.groups_size_sequential, got.work.inverses_pruned);
    }
}

/// Microbenchmark (ignored): one thread, production-shaped selections — the
/// outputs one own-route block consumes at 88.2 kHz / 180 fps — full inverse +
/// gather against the pruned inverse.
#[test]
#[ignore]
fn r10_micro_pruned_inverse() {
    let hop = 490usize;
    let mut seed = 0x1234_5678_9ABC_DEF1u64;
    for lg in 13..=16u32 {
        let n = 1usize << lg;
        let k = n / 2 - 100;
        let step = n - k + 1;
        let half = (k - 1) / 2;
        let inv = ASLT_PLANNER.with(|p| p.borrow_mut().plan_fft_inverse(n));
        let mut scratch = vec![Complex { re: 0.0f32, im: 0.0 }; inv.get_inplace_scratch_len()];
        let x = noise(n, &mut seed);
        // Frames owned by a block at base 0: positions fi*hop + half in [k-1, n).
        let idx: Vec<usize> = (0..).map(|fi| fi * hop + half).filter(|&p| p >= k - 1).take_while(|&p| p < n).collect();
        let mut w = work_for(&*inv, idx.len());
        let reps = (1usize << 24) / n;
        let mut out = vec![Complex { re: 0.0f32, im: 0.0 }; idx.len()];
        let mut buf = x.clone();
        let t = std::time::Instant::now();
        for _ in 0..reps {
            buf.copy_from_slice(&x);
            inv.process_with_scratch(&mut buf, &mut scratch);
            for (o, &i) in out.iter_mut().zip(&idx) { *o = buf[i]; }
        }
        let full = t.elapsed().as_secs_f64() / reps as f64;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            buf.copy_from_slice(&x);
            inv.process_pruned_with_scratch(&mut buf, &mut scratch, &idx, &mut out, pw(&mut w));
        }
        let pruned = t.elapsed().as_secs_f64() / reps as f64;
        let t = std::time::Instant::now();
        for _ in 0..reps { buf.copy_from_slice(&x); }
        let copy = t.elapsed().as_secs_f64() / reps as f64;
        // Product then pruned, against product fused into the first pass.
        let h = noise(n, &mut seed);
        let t = std::time::Instant::now();
        for _ in 0..reps {
            for ((o, a), hh) in buf.iter_mut().zip(&x).zip(&h) { *o = *a; *o *= *hh; }
            inv.process_pruned_with_scratch(&mut buf, &mut scratch, &idx, &mut out, pw(&mut w));
        }
        let sep = t.elapsed().as_secs_f64() / reps as f64;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            inv.process_pruned_product_with_scratch(&x, &h, &mut buf, &mut scratch, &idx, &mut out, pw(&mut w));
        }
        let fused = t.elapsed().as_secs_f64() / reps as f64;
        println!("   product+pruned {:.1} us, fused {:.1} us", sep * 1e6, fused * 1e6);
        println!("n 2^{lg}: step {step}, {} outputs; full {:.1} us, pruned {:.1} us, copy {:.1} us -> pruned/full (net of copy) {:.3}",
                 idx.len(), full * 1e6, pruned * 1e6, copy * 1e6, (pruned - copy) / (full - copy));
    }
}

/// Census (ignored): own-route members of the owner's descriptor and how many
/// share an exact `(n, k)` — within one member index of a 64-bar batch (what
/// the shared pass groups), across member indices within a batch, and across
/// the whole bar set.
#[test]
#[ignore]
fn r10_geometry_collisions() {
    use std::collections::HashMap;
    let cfg = AsltConfig { q_ratio: 2.0, max_window_s: 0.25, n_wavelets: 9, spread: 0.5,
                           scale: super::super::freq_scale::FreqScale::Blend(0.5) };
    for sr in [44_100u32, 88_200, 96_000] {
        let hop = hop_for_fps(sr, 180.0);
        let a_max = (sr as f32 / 2.0).min(24_000.0);
        let secs = 60.0f64;
        let n_sig = (secs * sr as f64) as usize;
        let frames = frame_count(n_sig, hop);
        let mut members: Vec<(usize, usize, usize, usize)> = Vec::new(); // bar, idx, n, k
        let mut direct = 0usize;
        for bar in 0..1024 {
            let f = cfg.scale.bar_center(bar, 1024, 20.0, a_max);
            if f >= sr as f32 / 2.0 { continue; }
            let q = effective_q_at(f, cfg.scale.grid_q_at(bar, 1024, 20.0, a_max), &cfg);
            for (i, &c) in superlet_cycles(q, &cfg).iter().enumerate() {
                let k = morlet_taps(f, c, sr as f32);
                if own_route_applies(k, n_sig, hop, frames) { members.push((bar, i, fft_block_len(k), k)); } else { direct += 1; }
            }
        }
        let mut within: HashMap<(usize, usize, usize, usize), usize> = HashMap::new();
        let mut across: HashMap<(usize, usize, usize), usize> = HashMap::new();
        let mut global: HashMap<(usize, usize), usize> = HashMap::new();
        for &(bar, i, n, k) in &members {
            *within.entry((bar / 64, i, n, k)).or_default() += 1;
            *across.entry((bar / 64, n, k)).or_default() += 1;
            *global.entry((n, k)).or_default() += 1;
        }
        let shared_within: usize = within.values().filter(|&&c| c >= 4).sum();
        let pair_across: usize = across.values().filter(|&&c| c >= 2).sum();
        let ge4_across: usize = across.values().filter(|&&c| c >= 4).sum();
        let pair_global: usize = global.values().filter(|&&c| c >= 2).sum();
        println!("sr {sr}: own-route members {} (direct {direct}); shared now (>=4 same index/batch) {shared_within}; \
                  rest {}; of rest, in a batch (n,k) group >=2 across indices: {}, >=4: {}; global groups >=2: {}",
                 members.len(), members.len() - shared_within,
                 members.iter().filter(|m| within[&(m.0 / 64, m.1, m.2, m.3)] < 4 && across[&(m.0 / 64, m.2, m.3)] >= 2).count(),
                 members.iter().filter(|m| within[&(m.0 / 64, m.1, m.2, m.3)] < 4 && across[&(m.0 / 64, m.2, m.3)] >= 4).count(),
                 members.iter().filter(|m| within[&(m.0 / 64, m.1, m.2, m.3)] < 4 && global[&(m.2, m.3)] >= 2).count());
        let _ = (pair_across, ge4_across, pair_global);
    }
}

/// The pruned inverse (with the shared pass's member scheduling) under the
/// lifecycle the production pass guarantees: cancelled after a known shared
/// window, no further window or subbatch starts, nothing is published, every
/// sharing reservation is released; and every shared window's work ran on the
/// analysis pool (6 workers) it was installed on.
#[test]
fn the_pruned_path_cancels_cleanly_and_stays_on_its_pool() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;
    let sr = 44_100u32;
    let n = (sr as f32 * 2.5) as usize;
    let l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.031).sin() * 0.6).collect();
    let r: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).cos() * 0.4).collect();
    let cfg = AsltConfig { q_ratio: 2.0, max_window_s: 0.25, n_wavelets: 9, spread: 0.5,
                           scale: super::super::freq_scale::FreqScale::Blend(0.5) };
    let hop = hop_for_fps(sr, 180.0);
    fn never(_: usize) -> bool { false }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(6).thread_name(|i| format!("r10-pool-{i}")).build().unwrap();

    // Uncancelled: identical to the reference, every window on the pool.
    let names: Mutex<std::collections::BTreeSet<String>> = Mutex::new(Default::default());
    let note = |e: PassEvent| if matches!(e, PassEvent::SharedWindow { .. }) {
        names.lock().unwrap().insert(std::thread::current().name().unwrap_or("?").to_string());
    };
    let run = |seams: Seams, go: &(dyn Fn() -> bool + Sync)| pool.install(|| analyze_routed(
        Input::Joint { left: &l, right: &r }, sr, 48, 20.0, 22_050.0, hop, &cfg, go, &|_| {}, never, seams));
    let want = run(Seams { shared_budget: 0, full_inverse: true, ..Seams::production() }, &|| true);
    let got = run(Seams { shared_budget: 0, pass_event: Some(&note), ..Seams::production() }, &|| true);
    if avx() { assert!(got.work.inverses_pruned > 0, "setup: the pruned path did not run: {:?}", got.work); }
    let b = |p: &[Vec<f32>]| p.iter().map(|r| r.iter().map(|v| v.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>();
    assert!(got.work.shared_members > 0, "setup: nothing shared");
    assert_eq!((b(&got.mix), b(&got.left), b(&got.right)), (b(&want.mix), b(&want.left), b(&want.right)));
    let names = names.into_inner().unwrap();
    assert!(!names.is_empty() && names.iter().all(|n| n.starts_with("r10-pool-")), "windows ran off the pool: {names:?}");
    assert!(names.len() <= 6);

    // Cancelled after the second completed window.
    let probe = ShareBudget::new(SHARE_GEOMETRY.budget_bytes);
    let stop = AtomicBool::new(false);
    let (windows, after, prepared_after) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
    let hook = |e: PassEvent| match e {
        PassEvent::SharedWindow { .. } => {
            if stop.load(Ordering::SeqCst) { after.fetch_add(1, Ordering::SeqCst); }
            if windows.fetch_add(1, Ordering::SeqCst) == 1 { stop.store(true, Ordering::SeqCst); }
        }
        PassEvent::SharedPrepared { .. } if stop.load(Ordering::SeqCst) => { prepared_after.fetch_add(1, Ordering::SeqCst); }
        _ => {}
    };
    let out = run(Seams { shared_budget: 0, share_probe: Some(&probe), pass_event: Some(&hook), ..Seams::production() },
                  &|| !stop.load(Ordering::SeqCst));
    assert!(windows.load(Ordering::SeqCst) >= 2, "setup: never reached the second window");
    assert_eq!(after.load(Ordering::SeqCst), 0, "a window ran after cancellation");
    assert_eq!(prepared_after.load(Ordering::SeqCst), 0, "a subbatch started after cancellation");
    assert!(out.mix.is_empty() && out.left.is_empty() && out.right.is_empty(), "a cancelled run published data");
    assert!(probe.peak() > 0, "setup: nothing reserved");
    assert_eq!(probe.used(), 0, "a reservation outlived the cancelled pass");
}

/// What the slot test's hook and its blockers share.
#[derive(Default)]
struct SlotWatch {
    m: std::sync::Mutex<SlotState>,
    cv: std::sync::Condvar,
}

#[derive(Default)]
struct SlotState {
    /// `(window, task, thread, created, resident)` as each worker saw it.
    runs: Vec<(usize, usize, usize, bool, usize)>,
    /// Members started per window, for the in-window rendezvous.
    started: std::collections::BTreeMap<usize, usize>,
    charge: Option<(usize, usize)>,
    released: Option<(usize, usize)>,
    coordinator: Option<usize>,
    /// Workers to hold busy before the next window, those held, and whether
    /// to let them go.
    targets: std::collections::BTreeSet<usize>,
    parked: std::collections::BTreeSet<usize>,
    release: bool,
    timed_out: bool,
}

/// Holds the pool thread it lands on if that thread is a target not yet
/// held; otherwise passes itself on to another thread while targets remain.
fn slot_blocker(w: std::sync::Arc<SlotWatch>) {
    let t = rayon::current_thread_index().expect("a blocker off the pool");
    let mut g = w.m.lock().unwrap();
    if g.release { return; }
    if g.targets.contains(&t) && !g.parked.contains(&t) {
        g.parked.insert(t);
        w.cv.notify_all();
        while !g.release { g = w.cv.wait(g).unwrap(); }
        return;
    }
    if g.parked.len() < g.targets.len() {
        drop(g);
        let w2 = w.clone();
        rayon::spawn(move || slot_blocker(w2));
    }
}

/// A shared subbatch's worker slots under real workers. Fewer members (2)
/// than pool workers (4), one block per window; between windows the workers
/// that have already run a member are held busy, and within a window each
/// member waits for the other to start, so each window's second member is
/// taken by a worker that has not run one before — until every worker has.
/// Observed by the workers themselves (`SlotRun`): which thread ran which
/// member, which slots were made then, and what each held. Then: no more
/// slots were made than the reservation charged for, their bytes are within
/// the charge, and every one is dropped when the subbatch ends — finished, and
/// cancelled after a known window.
#[test]
fn shared_subbatch_slots_stay_within_their_charge_whichever_workers_run_them() {
    use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
    use std::sync::Arc;
    use std::time::Duration;
    const T: usize = 4;
    let sr = 44_100.0f32;
    let len = 3 * 44_100;
    let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.013).sin() * 0.5).collect();
    let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.007).cos() * 0.5).collect();
    let hop = 245usize;
    let frames = frame_count(len, hop);
    let (freq, cycles) = (100.0f32, 5.0f32);
    let k = morlet_taps(freq, cycles, sr);
    let n = fft_block_len(k);
    assert!(own_route_applies(k, len, hop, frames), "setup: not an own-route geometry (k {k})");
    let plans: Vec<SizeSeqBar> = (0..2).map(|bar| SizeSeqBar { bar, freq, cycles: vec![cycles], passes: vec![0] }).collect();
    let windows = len.div_ceil(n - k + 1);
    assert!(windows > T + 2, "setup: {windows} windows");

    for cancel_after in [None, Some(2usize)] {
        let watch = Arc::new(SlotWatch::default());
        let stop = AtomicBool::new(false);
        let stopr = &stop;
        let w = watch.clone();
        let hook = move |e: PassEvent| {
            let me = rayon::current_thread_index().expect("an event off the pool");
            match e {
                PassEvent::SlotCharge { slots, bytes, .. } => {
                    let mut g = w.m.lock().unwrap();
                    g.charge = Some((slots, bytes));
                    g.coordinator = Some(me);
                }
                PassEvent::SlotRun { window, task, thread, created, resident, .. } => {
                    let mut g = w.m.lock().unwrap();
                    g.runs.push((window, task, thread, created, resident));
                    *g.started.entry(window).or_default() += 1;
                    w.cv.notify_all();
                    let deadline = std::time::Instant::now() + Duration::from_secs(20);
                    while g.started[&window] < 2 && !g.release {
                        let left = deadline.saturating_duration_since(std::time::Instant::now());
                        if left.is_zero() { g.timed_out = true; break; }
                        g = w.cv.wait_timeout(g, left).unwrap().0;
                    }
                }
                PassEvent::SharedWindow { window, .. } => {
                    if Some(window) == cancel_after { stopr.store(true, SeqCst); }
                    // Hold every worker but this one that has run a member,
                    // while one is left free to take the next window's second.
                    let mut g = w.m.lock().unwrap();
                    let used: std::collections::BTreeSet<usize> =
                        g.runs.iter().map(|r| r.2).filter(|&t| t != me).collect();
                    if g.release || used.len() >= T - 1 {
                        g.release = true;
                        w.cv.notify_all();
                        return;
                    }
                    g.targets = used;
                    let missing = g.targets.len() - g.parked.len();
                    drop(g);
                    for _ in 0..missing {
                        let w2 = w.clone();
                        rayon::spawn(move || slot_blocker(w2));
                    }
                    let mut g = w.m.lock().unwrap();
                    let deadline = std::time::Instant::now() + Duration::from_secs(20);
                    while g.parked.len() < g.targets.len() {
                        let left = deadline.saturating_duration_since(std::time::Instant::now());
                        if left.is_zero() { g.timed_out = true; break; }
                        g = w.cv.wait_timeout(g, left).unwrap().0;
                    }
                }
                PassEvent::SlotsReleased { created, released, .. } => {
                    let mut g = w.m.lock().unwrap();
                    g.released = Some((created, released));
                    g.release = true;
                    w.cv.notify_all();
                }
                _ => {}
            }
        };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(T).build().unwrap();
        let ok = pool.install(|| {
            let chans = Chans { a: &a, b: Some(&b), mean: None };
            let work = WorkCells::default();
            let seams = Seams { pass_event: Some(&hook), ..Seams::production() };
            let budget = ShareBudget::new(usize::MAX / 4);
            let halt = || stop.load(SeqCst);
            let ctx = ShareCtx {
                chans: &chans, hop, frames, sr, work: &work, seams: &seams, budget: &budget,
                cfg: ShareGeometry { budget_bytes: usize::MAX / 4, min_members: 1, window_blocks: 1, bars_per_batch: 64 },
                threads: rayon::current_num_threads(), stop: &halt,
            };
            let mut accs: Vec<Cols> = (0..2)
                .map(|_| Cols { mix: vec![0.0; frames], left: vec![0.0; frames], right: vec![0.0; frames] })
                .collect();
            let mut ms: Vec<MemberSlot> = accs.iter_mut().zip(&plans).collect();
            let ok = shared_geometry((n, k), &mut ms, 0, &ctx, &SharedSet::default());
            assert_eq!(budget.used(), 0, "a reservation outlived the subbatch");
            ok
        });
        {
            let mut g = watch.m.lock().unwrap();
            g.release = true;
            watch.cv.notify_all();
        }
        let g = watch.m.lock().unwrap();
        let what = if cancel_after.is_some() { "cancelled" } else { "finished" };
        assert!(!g.timed_out, "{what}: the schedule could not be forced");
        assert_eq!(ok, cancel_after.is_none(), "{what}: the subbatch's result");
        let (slots, bytes) = g.charge.expect("no charge seen");
        assert_eq!(slots, 2, "setup: the charge is for {slots} slots");
        let threads: std::collections::BTreeSet<usize> = g.runs.iter().map(|r| r.2).collect();
        let made: Vec<usize> = g.runs.iter().filter(|r| r.3).map(|r| r.4).collect();
        if cancel_after.is_none() {
            assert_eq!(threads.len(), T, "setup: members ran on {threads:?}, not on every worker");
            assert_eq!(g.runs.len(), 2 * windows, "every member ran every window once");
        } else {
            assert!(g.runs.iter().all(|r| r.0 <= 2), "a member ran after the cancelled window");
            assert!(threads.len() > 2, "setup: members ran on {threads:?} only");
        }
        assert!(made.len() <= slots, "{what}: {} slots made for a charge of {slots}, members on threads {threads:?}", made.len());
        assert!(made.iter().sum::<usize>() <= bytes,
                "{what}: slots held {} bytes for a charge of {bytes}", made.iter().sum::<usize>());
        assert_eq!(g.released, Some((made.len(), made.len())), "{what}: slots made vs dropped at the end");
    }
}

//! The exact GPU lane held to the CPU route (1.6.1). The device tests are
//! ignored: they need an NVIDIA sm_75 device, and they switch the process-wide
//! lane and its test faults, so they run on their own:
//! `cargo test --release exact_gpu -- --ignored --test-threads=1`.

use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn cfg() -> AsltConfig {
    AsltConfig { q_ratio: 2.0, max_window_s: 0.25, n_wavelets: 9, spread: 0.5,
                 scale: super::super::freq_scale::FreqScale::Blend(0.5) }
}

/// A generated stereo signal: partials a little apart in the two channels,
/// a sweep and noise; never silent.
fn signal(secs: f64, sr: u32) -> (Vec<f32>, Vec<f32>) {
    let n = (secs * sr as f64) as usize;
    let mut state = 0x2026_1104_9e37_79b9_u64;
    let mut noise = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let mut ch = |c: usize| -> Vec<f32> {
        (0..n).map(|i| {
            let t = i as f32 / sr as f32;
            let mut x = 0.0f32;
            for (j, f) in [55.0f32, 110.0, 220.0, 440.0, 880.0, 1760.0, 3520.0, 7040.0].iter().enumerate() {
                x += 0.5 / (1.0 + j as f32) * (std::f32::consts::TAU * f * (1.0 + 0.001 * c as f32) * t).sin();
            }
            x += 0.3 * (std::f32::consts::TAU * (40.0 + 20.0 * t / secs as f32) * t).sin();
            0.2 * x + 0.05 * noise()
        }).collect()
    };
    (ch(0), ch(1))
}

/// One analysis at the owner's descriptor (n 9, q 2, 1024 bars, 180 fps,
/// Blend 0.5, window 0.25 s) on six workers, the lane on or off.
fn analyse(l: &[f32], r: &[f32], sr: u32, lane: bool, go: &(dyn Fn() -> bool + Sync), done: &(dyn Fn(usize) + Sync)) -> Analysed {
    analyse_on(6, Input::Joint { left: l, right: r }, sr, lane, go, done)
}

fn analyse_on(workers: usize, input: Input, sr: u32, lane: bool, go: &(dyn Fn() -> bool + Sync), done: &(dyn Fn(usize) + Sync)) -> Analysed {
    analyse_seams(workers, input, sr, lane, go, done, Seams::production())
}

fn analyse_seams(workers: usize, input: Input, sr: u32, lane: bool, go: &(dyn Fn() -> bool + Sync),
                 done: &(dyn Fn(usize) + Sync), seams: Seams) -> Analysed {
    set_exact_gpu(lane);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap();
    let hop = hop_for_fps(sr, 180.0);
    let a_max = (sr as f32 / 2.0).min(24_000.0);
    let out = pool.install(|| analyze_routed(
        input, sr, 1024, 20.0, a_max, hop, &cfg(), go, done, route_to_device, seams,
    ));
    set_exact_gpu(false);
    out
}

fn same(a: &Analysed, b: &Analysed, what: &str) {
    for (name, x, y) in [("mix", &a.mix, &b.mix), ("left", &a.left, &b.left), ("right", &a.right, &b.right)] {
        assert_eq!(x.len(), y.len(), "{what}: {name} bars");
        for (bar, (u, v)) in x.iter().zip(y).enumerate() {
            assert_eq!(u.len(), v.len(), "{what}: {name} bar {bar} frames");
            if let Some(f) = (0..u.len()).find(|&f| u[f].to_bits() != v[f].to_bits()) {
                panic!("{what}: {name} bar {bar} frame {f}: {} vs {}", u[f], v[f]);
            }
        }
    }
}

/// The 2^15 group's plan, as `analyze_routed` makes it.
fn plan_2_15(n_bars: usize, sr: u32, n_sig: usize, hop: usize) -> Vec<SizeSeqBar> {
    let cfg = cfg();
    let srf = sr as f32;
    let (lo, hi) = (20.0f32, (srf / 2.0).min(24_000.0));
    let bars: Vec<(usize, f32, f32)> = (0..n_bars).filter_map(|bar| {
        let f = cfg.scale.bar_center(bar, n_bars, lo, hi);
        if f >= srf / 2.0 { return None; }
        let q = effective_q_at(f, cfg.scale.grid_q_at(bar, n_bars, lo, hi), &cfg);
        let max_n = superlet_cycles(q, &cfg).into_iter().map(|c| morlet_taps(f, c, srf))
            .filter(|&k| fft_is_cheaper(k, hop)).map(fft_block_len).filter(|&n| n > 0).max();
        (max_n == Some(exact_gpu::N)).then_some((bar, f, q))
    }).collect();
    let _ = n_sig;
    plan_size_sequential(&bars, &[exact_gpu::N], hop, &cfg, srf).expect("a size-sequential plan")
}

/// A bank's columns against `member_cols` on the CPU, for the first and last
/// member position of two chunks, at 48 and 44.1 kHz: every value of every
/// plane, bit for bit.
#[test]
#[ignore]
fn a_bank_is_the_cpu_routes_members_bit_for_bit() {
    let _calib = super::super::gpu_calib::isolated_test_env("exactgpu-bank");
    let lane = exact_gpu::Lane::start().expect("an NVIDIA sm_75 device");
    for sr in [48_000u32, 44_100] {
        let (l, r) = signal(20.0, sr);
        let hop = hop_for_fps(sr, 180.0);
        let frames = frame_count(l.len(), hop);
        let plans = plan_2_15(1024, sr, l.len(), hop);
        let (chunks, rest) = exact_gpu_chunks(&plans, exact_gpu::N, l.len(), hop, frames, sr as f32);
        assert!(chunks.len() >= 2, "{sr}: chunks {} rest {}", chunks.len(), rest.len());
        let mut set = SharedSet::default();
        set.a.insert(exact_gpu::N, Arc::new(SignalBlocks::build(&l, exact_gpu::N)));
        set.b.insert(exact_gpu::N, Arc::new(SignalBlocks::build(&r, exact_gpu::N)));
        let chans = Chans { a: &l, b: Some(&r), mean: None };
        let (work, seams) = (WorkCells::default(), Seams::production());
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        for chunk in [&chunks[0], chunks.last().unwrap()] {
            let bars: Vec<&SizeSeqBar> = chunk.iter().map(|&i| &plans[i]).collect();
            let positions = pass_cycles(bars[0], exact_gpu::N).count();
            for q in [0, positions - 1] {
                let got = exact_gpu_bank(&lane, &cancel, &|| false, true, &bars, q, exact_gpu::N, &chans, hop, frames, sr as f32, &set, &work, &seams)
                    .unwrap_or_else(|e| panic!("{sr} q {q}: {}", match e { BankFault::Failed(e) => e, BankFault::Cancelled => "cancelled".into() }));
                let taps = morlet_taps(bars[0].freq, pass_cycles(bars[0], exact_gpu::N).nth(q).unwrap(), sr as f32);
                for (p, g) in bars.iter().zip(&got) {
                    let c = pass_cycles(p, exact_gpu::N).nth(q).unwrap();
                    let want = Morlet::new(p.freq, c, sr as f32).member_cols(&chans, hop, frames, &set, &work, &seams);
                    for (name, x, y) in [("mix", &g.mix, &want.mix), ("left", &g.left, &want.left), ("right", &g.right, &want.right)] {
                        assert_eq!(x.len(), y.len());
                        if let Some(f) = (0..x.len()).find(|&f| x[f].to_bits() != y[f].to_bits()) {
                            panic!("{sr} taps {taps} bar {} {name} frame {f}: {} vs {}", p.bar, x[f], y[f]);
                        }
                    }
                }
                eprintln!("EXACT {sr} Hz taps {taps}: 32 members x 3 planes x {frames} frames identical");
            }
        }
        let w = WorkCount::snapshot(&work);
        assert_eq!(w.exact_gpu_banks, 4);
        assert_eq!(w.exact_gpu_failures, 0);
    }
}

/// A whole joint analysis with the lane against one without: identical
/// planes, and the lane did take work.
#[test]
#[ignore]
fn an_analysis_with_the_lane_is_the_analysis_without_it() {
    let _calib = super::super::gpu_calib::isolated_test_env("exactgpu-whole");
    for sr in [48_000u32, 44_100] {
        let (l, r) = signal(30.0, sr);
        let off = analyse(&l, &r, sr, false, &|| true, &|_| {});
        let on = analyse(&l, &r, sr, true, &|| true, &|_| {});
        assert_eq!(off.work.exact_gpu_members, 0);
        assert!(on.work.exact_gpu_members > 0, "the lane took nothing");
        assert_eq!(on.work.exact_gpu_failures, 0);
        same(&on, &off, &format!("{sr} Hz"));
        eprintln!("EXACT whole {sr} Hz: {} members, {} banks, {} device blocks; planes identical",
                  on.work.exact_gpu_members, on.work.exact_gpu_banks, on.work.exact_gpu_blocks);
    }
}

/// With two workers (one for the lane, one core) the same; a mono analysis
/// does not use the lane at all and is unchanged.
#[test]
#[ignore]
fn two_workers_and_mono_analyses_are_unchanged() {
    let _calib = super::super::gpu_calib::isolated_test_env("exactgpu-two-mono");
    let sr = 44_100;
    let (l, r) = signal(20.0, sr);
    let off = analyse_on(2, Input::Joint { left: &l, right: &r }, sr, false, &|| true, &|_| {});
    let on = analyse_on(2, Input::Joint { left: &l, right: &r }, sr, true, &|| true, &|_| {});
    assert!(on.work.exact_gpu_members > 0, "the lane took nothing with two workers");
    same(&on, &off, "two workers");
    let off = analyse_on(6, Input::Mono(&l), sr, false, &|| true, &|_| {});
    let on = analyse_on(6, Input::Mono(&l), sr, true, &|| true, &|_| {});
    assert_eq!(on.work.exact_gpu_members, 0, "a mono analysis used the lane");
    same(&on, &off, "mono");
}

/// A bank that fails part-way is recomputed on the cores and the lane is
/// given up; the analysis is unchanged.
#[test]
#[ignore]
fn a_failed_bank_is_recomputed_on_the_cores() {
    let _calib = super::super::gpu_calib::isolated_test_env("exactgpu-fail");
    let sr = 48_000;
    let (l, r) = signal(30.0, sr);
    let off = analyse(&l, &r, sr, false, &|| true, &|_| {});
    // Part-way through the second bank.
    let blocks = exact_gpu::fault::BLOCKS.load(Ordering::SeqCst);
    exact_gpu::fault::FAIL_AT.store(blocks + 150, Ordering::SeqCst);
    let on = analyse(&l, &r, sr, true, &|| true, &|_| {});
    exact_gpu::fault::FAIL_AT.store(usize::MAX, Ordering::SeqCst);
    assert_eq!(on.work.exact_gpu_failures, 1);
    assert!(on.work.exact_gpu_replayed > 0);
    assert!(on.work.exact_gpu_banks >= 1, "the first bank completed");
    same(&on, &off, "after a failed bank");
}

/// A device that returns one wrong bit is caught by the check before its
/// bank is used; the analysis is unchanged.
#[test]
#[ignore]
fn a_bank_the_check_rejects_is_recomputed_on_the_cores() {
    let _calib = super::super::gpu_calib::isolated_test_env("exactgpu-check");
    let sr = 44_100;
    let (l, r) = signal(30.0, sr);
    let off = analyse(&l, &r, sr, false, &|| true, &|_| {});
    exact_gpu::fault::CORRUPT.store(true, Ordering::SeqCst);
    let on = analyse(&l, &r, sr, true, &|| true, &|_| {});
    exact_gpu::fault::CORRUPT.store(false, Ordering::SeqCst);
    assert_eq!(on.work.exact_gpu_failures, 1);
    assert_eq!(on.work.exact_gpu_banks, 0, "the rejected bank was never counted as done");
    same(&on, &off, "after a rejected check");
}

/// Cancelled while the lane works: nothing comes back, the lane is retired,
/// and the next analysis runs whole and right.
#[test]
#[ignore]
fn a_cancelled_analysis_returns_nothing_and_the_next_is_whole() {
    let _calib = super::super::gpu_calib::isolated_test_env("exactgpu-cancel");
    let sr = 48_000;
    let (l, r) = signal(30.0, sr);
    // Cancel once a few bars of the 2^15 group have finished, which happens
    // only inside its 2^15 pass (its last), with the lane running.
    let bars = AtomicUsize::new(0);
    let go = || bars.load(Ordering::SeqCst) < 40;
    let done = |_: usize| { bars.fetch_add(1, Ordering::SeqCst); };
    let t0 = std::time::Instant::now();
    let cancelled = analyse(&l, &r, sr, true, &go, &done);
    eprintln!("EXACT cancelled after {} bars in {:.2} s (a cancelled analysis reports no work counts)",
              bars.load(Ordering::SeqCst), t0.elapsed().as_secs_f64());
    assert!(cancelled.mix.iter().all(|c| c.is_empty()) || cancelled.mix.is_empty(), "a cancelled analysis returned columns");
    let off = analyse(&l, &r, sr, false, &|| true, &|_| {});
    let on = analyse(&l, &r, sr, true, &|| true, &|_| {});
    assert!(on.work.exact_gpu_members > 0);
    same(&on, &off, "after a cancelled one");
}

/// What the lane costs in process memory, by stage: commit before it starts,
/// after its session is up, after one bank, and after it is dropped.
#[test]
#[ignore]
fn what_the_lane_holds() {
    let mem = || super::r10m::probe::mem();
    let mib = |b: u64| b as f64 / 1048576.0;
    let (c0, _, w0, _) = mem();
    let t0 = std::time::Instant::now();
    let lane = exact_gpu::Lane::start().expect("an NVIDIA sm_75 device");
    let setup = t0.elapsed();
    let (c1, _, w1, _) = mem();
    let sr = 48_000u32;
    let (l, r) = signal(20.0, sr);
    let hop = hop_for_fps(sr, 180.0);
    let frames = frame_count(l.len(), hop);
    let plans = plan_2_15(1024, sr, l.len(), hop);
    let (chunks, _) = exact_gpu_chunks(&plans, exact_gpu::N, l.len(), hop, frames, sr as f32);
    let mut set = SharedSet::default();
    set.a.insert(exact_gpu::N, Arc::new(SignalBlocks::build(&l, exact_gpu::N)));
    set.b.insert(exact_gpu::N, Arc::new(SignalBlocks::build(&r, exact_gpu::N)));
    let chans = Chans { a: &l, b: Some(&r), mean: None };
    let (work, seams) = (WorkCells::default(), Seams::production());
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (c2, _, w2, _) = mem();
    let bars: Vec<&SizeSeqBar> = chunks[0].iter().map(|&i| &plans[i]).collect();
    let cols = exact_gpu_bank(&lane, &cancel, &|| false, false, &bars, 0, exact_gpu::N, &chans, hop, frames, sr as f32, &set, &work, &seams)
        .ok().expect("a bank");
    let (c3, peak3, w3, _) = mem();
    drop(cols);
    lane.retire().expect("a clean retirement");
    drop(bars);
    drop((set, l, r, plans, chunks));
    let (c4, _, w4, _) = mem();
    eprintln!("LANEMEM setup {:.0} ms; commit MiB: before {:.1}, lane up +{:.1}, inputs +{:.1}, after a bank +{:.1} (peak {:.1}); everything dropped: {:+.1} vs before; \
               working set MiB: {:.1} / {:.1} / {:.1} / {:.1} / {:.1}; device buffers {:.1} MiB",
              setup.as_secs_f64() * 1e3, mib(c0), mib(c1 - c0), mib(c2 - c1), mib(c3 - c2), mib(peak3), mib(c4) - mib(c0),
              mib(w0), mib(w1), mib(w2), mib(w3), mib(w4), exact_gpu::DEVICE_BYTES as f64 / 1048576.0);
}

/// Clears every test fault and opens every device gate however a test ends,
/// so a failing test leaves no fault armed and no lane thread held.
struct Gates;

impl Drop for Gates {
    fn drop(&mut self) {
        use exact_gpu::fault;
        fault::HOLD_AT.store(usize::MAX, Ordering::SeqCst);
        fault::RELEASE.store(true, Ordering::SeqCst);
        fault::FAIL_AT.store(usize::MAX, Ordering::SeqCst);
        fault::CORRUPT.store(false, Ordering::SeqCst);
        fault::FAIL_CLEANUP.store(false, Ordering::SeqCst);
        fault::PANIC.store(0, Ordering::SeqCst);
    }
}

/// Waits up to `secs` for `cond`.
fn wait_for(secs: u64, cond: impl Fn() -> bool) -> bool {
    let t = std::time::Instant::now();
    while !cond() {
        if t.elapsed() > std::time::Duration::from_secs(secs) {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    true
}

/// Cancelled while a bank is on the device and the cores have nothing left
/// to do: the lane's wait for the device sees it, the device stops at its
/// next block, nothing more is reported done, and the analysis returns
/// nothing. The schedule is forced: the device is held before the sixth block
/// of its first bank until the cores have drained, the analysis is stopped,
/// and the wait has seen the stop.
#[test]
#[ignore]
fn a_cancel_after_the_cores_drain_stops_the_bank_in_flight() {
    use exact_gpu::fault;
    use std::sync::atomic::AtomicBool;
    let _calib = super::super::gpu_calib::isolated_test_env("exactgpu-cancel-drained");
    let _gates = Gates;
    let sr = 48_000;
    let (l, r) = signal(30.0, sr);
    fault::RELEASE.store(false, Ordering::SeqCst);
    fault::STOP_SEEN.store(false, Ordering::SeqCst);
    let held = fault::BLOCKS.load(Ordering::SeqCst) + 5;
    fault::HOLD_AT.store(held, Ordering::SeqCst);
    let (drained, stopped) = (AtomicBool::new(false), AtomicBool::new(false));
    let late = AtomicUsize::new(0);
    let hook = |e: PassEvent| if matches!(e, PassEvent::CoresDrained) { drained.store(true, Ordering::SeqCst) };
    let go = || !stopped.load(Ordering::SeqCst);
    let done = |_: usize| if stopped.load(Ordering::SeqCst) { late.fetch_add(1, Ordering::SeqCst); };
    let (out, schedule) = std::thread::scope(|s| {
        let conductor = s.spawn(|| {
            let _gates = Gates;
            if !wait_for(600, || drained.load(Ordering::SeqCst)) {
                return Err("the cores never drained");
            }
            if !wait_for(60, || fault::HOLDING.load(Ordering::SeqCst)) {
                return Err("the device was never held");
            }
            stopped.store(true, Ordering::SeqCst);
            if !wait_for(10, || fault::STOP_SEEN.load(Ordering::SeqCst)) {
                return Err("the wait for the device never saw the stop");
            }
            Ok(fault::BLOCKS.load(Ordering::SeqCst))
        });
        let out = analyse_seams(6, Input::Joint { left: &l, right: &r }, sr, true, &go, &done,
                                Seams { pass_event: Some(&hook), ..Seams::production() });
        (out, conductor.join().expect("the conductor"))
    });
    let released = schedule.unwrap_or_else(|e| panic!("schedule: {e}"));
    assert_eq!(released, held, "the device ran past its hold before it was released");
    // The held block and every one after it in the bank were never run.
    assert_eq!(fault::BLOCKS.load(Ordering::SeqCst), held, "the device ran blocks after the stop");
    assert_eq!(late.load(Ordering::SeqCst), 0, "bars were reported done after the stop");
    // (A cancelled analysis reports no work counts either.)
    assert!(out.mix.is_empty() && out.left.is_empty() && out.codes.is_none(), "a cancelled analysis returned a result");
    eprintln!("EXACT cancel after the cores drained: the device stopped at its held block {held}; nothing returned");
}

/// A long-window descriptor (0.75 s, members from 0.2 of the window up) with
/// a 2^17 group, one the legacy calibration routes, whose capped bars have
/// two members in the 2^15 pass that the lane takes.
fn cfg_long() -> AsltConfig {
    AsltConfig { max_window_s: 0.75, spread: 0.2, ..cfg() }
}

/// A group the legacy calibration routes in which the lane was asked to
/// start is not a cores sample, whatever became of the lane: here it fails
/// at its first block, its first bank fails the check, or its thread panics
/// while it is set up. No bank is done, the cores do the work, the analysis
/// is the cores-only one, and the calibration learns nothing from it.
#[test]
#[ignore]
fn a_group_the_lane_was_tried_in_is_not_a_cores_sample() {
    use super::super::gpu_calib::{self as calib, Outcome};
    use exact_gpu::fault;
    const GROUP: usize = 1 << 17;
    fn via_calibration(n: usize) -> bool {
        super::super::gpu_calib::use_device(n)
    }
    let _env = calib::aslt_test_env("exactgpu-calib");
    let _gates = Gates;
    let take = || std::mem::take(&mut *calib::RECORDED.lock().unwrap_or_else(|e| e.into_inner()));
    let cores = |rec: &[(usize, bool, Outcome)]| rec.iter().filter(|&&(n, dev, o)| n == GROUP && !dev && o == Outcome::Cores).count();
    let sr = 48_000u32;
    let (l, r) = signal(16.0, sr);
    let hop = hop_for_fps(sr, 180.0);
    let run = |lane: bool| {
        set_exact_gpu(lane);
        let out = calib::install(|| {
            let out = analyze_routed(Input::Joint { left: &l, right: &r }, sr, 1024, 20.0, 24_000.0, hop, &cfg_long(),
                                     &|| true, &|_| {}, via_calibration, Seams::production());
            (out, calib::pending_is_empty())
        });
        set_exact_gpu(false);
        out
    };
    take();
    let (off, clean) = run(false);
    let rec = take();
    assert!(clean);
    assert_eq!(off.work.exact_gpu_attempts, 0);
    assert_eq!(cores(&rec), 1, "setup: without the lane the 2^17 group is one cores sample: {rec:?}");
    let trained = calib::samples_for_test(GROUP);
    assert_eq!(trained.map(|(_, c)| c), Some(1), "setup: the cores sample was not measurable");
    let cases: [(&str, &dyn Fn()); 3] = [
        ("a failed first block", &|| fault::FAIL_AT.store(fault::BLOCKS.load(Ordering::SeqCst), Ordering::SeqCst)),
        ("a first bank the check rejects", &|| fault::CORRUPT.store(true, Ordering::SeqCst)),
        ("a lane that panics while set up", &|| fault::PANIC.store(1, Ordering::SeqCst)),
    ];
    for (what, arm) in cases {
        arm();
        let (on, clean) = run(true);
        let rec = take();
        fault::FAIL_AT.store(usize::MAX, Ordering::SeqCst);
        fault::CORRUPT.store(false, Ordering::SeqCst);
        fault::PANIC.store(0, Ordering::SeqCst);
        let w = &on.work;
        eprintln!("EXACT calibration, {what}: attempts {}, banks {}, failures {}, replayed {}; 2^17 records {:?}",
                  w.exact_gpu_attempts, w.exact_gpu_banks, w.exact_gpu_failures, w.exact_gpu_replayed,
                  rec.iter().filter(|&&(n, ..)| n == GROUP).collect::<Vec<_>>());
        assert!(clean, "{what}: a decision was left parked");
        assert_eq!(w.exact_gpu_attempts, 1, "{what}: the lane was not tried");
        assert_eq!((w.exact_gpu_banks, w.exact_gpu_members), (0, 0), "{what}: a bank was done");
        if what.contains("panics") {
            assert_eq!(w.exact_gpu_failures, 0);
        } else {
            assert_eq!(w.exact_gpu_failures, 1, "{what}");
            assert!(w.exact_gpu_replayed > 0, "{what}: nothing replayed on the cores");
        }
        same(&on, &off, what);
        assert_eq!(cores(&rec), 0, "{what}: recorded as a cores sample: {rec:?}");
        assert_eq!(calib::samples_for_test(GROUP), trained, "{what}: the calibration was trained");
    }
}

/// The lane's retirement reports what went wrong and still does the rest: a
/// failed cleanup step with every other step attempted, a thread that
/// panicked at a bank (its session released while it unwound), and one that
/// panicked while it was set up.
#[test]
#[ignore]
fn a_failed_cleanup_and_a_panicking_lane_are_reported() {
    use exact_gpu::fault;
    let _calib = super::super::gpu_calib::isolated_test_env("exactgpu-retire");
    let _gates = Gates;
    // The synchronise, five frees, the unload and the destroy.
    const STEPS: usize = 8;
    let steps = || fault::CLEANUP_STEPS.load(Ordering::SeqCst);

    let lane = exact_gpu::Lane::start().expect("an NVIDIA sm_75 device");
    let s0 = steps();
    lane.retire().expect("a clean retirement");
    assert_eq!(steps() - s0, STEPS);

    let lane = exact_gpu::Lane::start().expect("an NVIDIA sm_75 device");
    fault::FAIL_CLEANUP.store(true, Ordering::SeqCst);
    let s0 = steps();
    let e = lane.retire().expect_err("a failed free is reported");
    assert!(e.contains("cuMemFree"), "{e}");
    assert_eq!(steps() - s0, STEPS, "a step after the failed one was skipped");

    let lane = exact_gpu::Lane::start().expect("an NVIDIA sm_75 device");
    let (l, r) = signal(2.0, 48_000);
    let sets = [Arc::new(SignalBlocks::build(&l, exact_gpu::N)), Arc::new(SignalBlocks::build(&r, exact_gpu::N))];
    let bank = exact_gpu::Bank { h: Vec::new(), sets, hop: 267, blocks: Vec::new(), cols: Vec::new() };
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    fault::PANIC.store(2, Ordering::SeqCst);
    let s0 = steps();
    match lane.run(bank, &cancel, false, &|| false) {
        exact_gpu::Reply::Failed(e) => assert!(e.contains("without replying"), "{e}"),
        _ => panic!("a lane that panicked replied"),
    }
    let e = lane.retire().expect_err("a panic is reported");
    assert!(e.contains("panicked") && e.contains("forced lane panic"), "{e}");
    assert_eq!(steps() - s0, STEPS, "the unwinding thread did not release its session");

    fault::PANIC.store(1, Ordering::SeqCst);
    let e = exact_gpu::Lane::start().err().expect("a lane that panicked while set up");
    assert!(e.contains("panicked") && e.contains("forced lane panic"), "{e}");
    eprintln!("EXACT retirement: failed cleanup and panics reported, every cleanup step attempted");
}

/// The bank estimate bounds every kernel the pass can admit, and does not
/// overflow.
#[test]
fn the_bank_estimate_bounds_every_admitted_kernel() {
    let n = exact_gpu::N;
    // A 2^15 block is chosen for kernels of up to n/2 taps and no longer: the
    // estimate's bound on a wavelet.
    assert_eq!(fft_block_len(n / 2), n);
    assert!(fft_block_len(n / 2 + 1) > n);
    // The review's example, admitted with 64 outputs at the 48 kHz hop.
    let (k, hop) = (15_681, 267);
    let valid = n + 1 - k;
    assert!(valid / hop >= exact_gpu::MIN_M && valid.div_ceil(hop) <= exact_gpu::MAX_M);
    let frames = 34_200;
    let (c, f) = (std::mem::size_of::<C32>(), std::mem::size_of::<f32>());
    let held = exact_gpu::BANK * (3 * k * f + n * c + frames * (2 * c + 3 * f));
    let bound = exact_gpu_bank_bytes(frames).unwrap();
    assert!(held < bound, "{held} vs {bound}");
    assert!(exact_gpu_bank_bytes(usize::MAX / 8).is_none());
    // The longest track whose bank fits the budget.
    let longest = (0..).map(|m| m * 1000).take_while(|&fr| exact_gpu_bank_bytes(fr).is_some_and(|b| b <= EXACT_GPU_HOST_BUDGET)).last().unwrap();
    eprintln!("EXACT bank estimate: {:.1} MiB at 190 s; fits up to {} frames ({:.0} min at 180 fps)",
              bound as f64 / 1048576.0, longest, longest as f64 / 180.0 / 60.0);
}

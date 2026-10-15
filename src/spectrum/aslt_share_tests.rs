//! Forward-transform sharing in the size-sequential no-blocks pass
//! ([`run_shared_pass0`]): held to the member-by-member pass it replaces, bit
//! for bit, through the production selection, and checked for its budget,
//! progress and cancellation behaviour.

use super::*;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const SR: u32 = 44_100;
const BARS: usize = 48;

/// The owner's descriptor at 48 bars.
fn cfg() -> AsltConfig {
    AsltConfig {
        q_ratio: 2.0, max_window_s: 0.25, n_wavelets: 9, spread: 0.5,
        scale: super::super::freq_scale::FreqScale::Blend(0.5),
    }
}

fn signals(secs: f32) -> Vec<(&'static str, Vec<f32>, Vec<f32>)> {
    let n = (SR as f32 * secs) as usize;
    let tau = std::f32::consts::TAU;
    let tone = |f: f32, a: f32, ph: f32| -> Vec<f32> {
        (0..n).map(|i| a * (tau * f * i as f32 / SR as f32 + ph).sin()).collect()
    };
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut noise = || -> Vec<f32> {
        (0..n).map(|_| {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            ((seed >> 40) as f32 / 8_388_608.0) - 1.0
        }).collect()
    };
    let clicks: Vec<f32> = (0..n).map(|i| if i % (SR as usize / 3) == 7 { 0.9 } else { 0.0 }).collect();
    vec![
        ("noise", noise(), noise()),
        ("anti phase", tone(220.0, 0.6, 0.0), tone(220.0, 0.6, std::f32::consts::PI)),
        ("transients", clicks, tone(1500.0, 0.3, 0.2)),
        ("one silent", tone(90.0, 0.7, 0.0), vec![0.0; n]),
    ]
}

fn never_gpu(_: usize) -> bool { false }

fn run(input: Input, seams: Seams, go: &(dyn Fn() -> bool + Sync), done: &(dyn Fn(usize) + Sync)) -> Analysed {
    analyze_routed(input, SR, BARS, 20.0, 22_050.0, hop_for_fps(SR, 180.0), &cfg(), go, done, never_gpu, seams)
}

/// What the shared pass is held to: the same production seams with the
/// no-blocks pass computed member by member.
fn unshared(s: Seams) -> Seams { Seams { share_geometry: None, ..s } }

fn bits(p: &[Vec<f32>]) -> Vec<Vec<u32>> { p.iter().map(|r| r.iter().map(|v| v.to_bits()).collect()).collect() }

fn same(got: &Analysed, want: &Analysed, what: &str) {
    assert!(!want.mix.is_empty(), "{what}: setup: empty reference");
    assert_eq!(bits(&got.mix), bits(&want.mix), "{what}: mix");
    assert_eq!(bits(&got.left), bits(&want.left), "{what}: left");
    assert_eq!(bits(&got.right), bits(&want.right), "{what}: right");
}

/// Production's selection (`Seams::production()`), on a track whose sizes the
/// block budget refuses — the long-track case, made with the existing budget
/// seam — against the member-by-member pass: every bit of every plane, joint
/// and mono, with both the shared branch and the unchanged-route branch of the
/// pass witnessed by its counters. Codes too, through the cache's map.
#[test]
fn production_sharing_is_the_member_route_bit_for_bit() {
    assert!(Seams::production().share_geometry == Some(SHARE_GEOMETRY), "production does not select sharing");
    let own = || Seams { shared_budget: 0, ..Seams::production() };
    for (name, l, r) in signals(2.5) {
        let joint = Input::Joint { left: &l, right: &r };
        let want = run(Input::Joint { left: &l, right: &r }, unshared(own()), &|| true, &|_| {});
        let got = run(joint, own(), &|| true, &|_| {});
        same(&got, &want, name);
        let w = got.work;
        assert!(w.shared_members > 0, "{name}: the shared branch did not run");
        assert!(w.column_convolutions > 2 * w.shared_members,
                "{name}: no member took the unchanged route inside the pass ({w:?})");
        assert_eq!(w.share_refused_members, 0, "{name}: production budget refused at this size");
        assert_eq!(want.work.shared_members, 0, "{name}: the reference shared");
        assert_eq!(w.column_convolutions, want.work.column_convolutions, "{name}: different member work");

        let want = run(Input::Mono(&l), unshared(own()), &|| true, &|_| {});
        let got = run(Input::Mono(&l), own(), &|| true, &|_| {});
        same(&got, &want, &format!("{name} mono"));
        assert!(got.work.shared_members > 0, "{name} mono: the shared branch did not run");
    }
    let (_, l, r) = signals(2.5).swap_remove(0);
    let map = |m: f32| crate::spectrum::aslt_code(m);
    let codes = |s: Seams| run(Input::Joint { left: &l, right: &r }, Seams { codes: Some(&map), ..s }, &|| true, &|_| {})
        .codes.expect("codes");
    assert_eq!(codes(own()), codes(unshared(own())), "codes differ");
}

// The test's two-member geometries are a change from production's.
const _: () = assert!(SHARE_GEOMETRY.min_members > 2);

/// The budget: a reservation that does not fit splits the geometry into
/// smaller subbatches, one that cannot fit `min_members` sends the rest to the
/// unchanged route — and the values do not move either way. Reservations are
/// checked against the cap and are all released.
#[test]
fn a_small_budget_splits_and_refuses_without_changing_a_bit() {
    let (_, l, r) = signals(2.5).swap_remove(0);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    // Two-member geometries may share here, so 48 bars give geometries big
    // enough to split.
    let base = || Seams { shared_budget: 0, share_geometry: Some(ShareGeometry { min_members: 2, ..SHARE_GEOMETRY }),
                          ..Seams::production() };
    let want = pool.install(|| run(Input::Joint { left: &l, right: &r }, unshared(base()), &|| true, &|_| {}));
    let full = ShareBudget::new(SHARE_GEOMETRY.budget_bytes);
    let whole = pool.install(|| run(Input::Joint { left: &l, right: &r }, Seams { share_probe: Some(&full), ..base() }, &|| true, &|_| {}));
    same(&whole, &want, "production budget");
    assert!(full.peak() > 0 && full.peak() <= SHARE_GEOMETRY.budget_bytes, "peak {}", full.peak());
    assert_eq!(full.used(), 0, "a reservation outlived its subbatch");
    // Caps below the whole run's largest reservation: every one gives the same
    // bits within its cap, and at least one admits some geometry in more than
    // one subbatch (witnessed by its preparations).
    let mut split_seen = false;
    for tenths in [9usize, 8, 7, 6, 5, 4, 3] {
        let cap = full.peak() * tenths / 10;
        let small = ShareBudget::new(cap);
        let preps: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());
        let hook = |e: PassEvent| if let PassEvent::SharedPrepared { member, n, k, .. } = e {
            preps.lock().unwrap().push((member, n, k));
        };
        let split = pool.install(|| run(Input::Joint { left: &l, right: &r },
                                        Seams { share_probe: Some(&small), pass_event: Some(&hook), ..base() }, &|| true, &|_| {}));
        same(&split, &want, &format!("cap {tenths}/10"));
        assert!(small.peak() <= cap, "reserved {} over a cap of {cap}", small.peak());
        assert_eq!(small.used(), 0, "a reservation outlived its subbatch");
        let mut p = preps.into_inner().unwrap();
        let all = p.len();
        p.sort_unstable();
        p.dedup();
        split_seen |= p.len() < all && split.work.shared_members > 0;
    }
    assert!(split_seen, "no cap split a geometry into subbatches");
    // Too small for any subbatch: every candidate member refused.
    let none = ShareBudget::new(1);
    let refused = pool.install(|| run(Input::Joint { left: &l, right: &r }, Seams { share_probe: Some(&none), ..base() }, &|| true, &|_| {}));
    same(&refused, &want, "refused");
    assert_eq!(refused.work.shared_members, 0);
    assert!(refused.work.share_refused_members > 0, "setup: nothing was refused");
    assert_eq!(none.peak(), 0);
}

/// A group whose largest size the block budget admits and whose smaller
/// sizes it does not: shared blocks for one pass, the shared no-blocks pass
/// before it. Same bits; and every bar reported done exactly once, after the
/// last of its passes — never at the end of the no-blocks pass while a later
/// pass still has members for it.
#[test]
fn mixed_passes_keep_their_bits_and_report_each_bar_once_when_it_is_done() {
    let (_, l, r) = signals(2.5).swap_remove(0);
    // Exactly one joint block set fits.
    let one_size = 2 * 2 * l.len() * std::mem::size_of::<rustfft::num_complex::Complex<f32>>();
    #[derive(Clone, Debug)]
    enum Ev { P(PassEvent), Done(usize), Part(usize, f32) }
    let log: Mutex<Vec<Ev>> = Mutex::new(Vec::new());
    let ev = |e: PassEvent| log.lock().unwrap().push(Ev::P(e));
    let done = |b: usize| log.lock().unwrap().push(Ev::Done(b));
    let part = |b: usize, f: f32| log.lock().unwrap().push(Ev::Part(b, f));
    let small_batches = ShareGeometry { bars_per_batch: 5, ..SHARE_GEOMETRY };
    let seams = || Seams { shared_budget: one_size, ..Seams::production() };
    let want = run(Input::Joint { left: &l, right: &r }, unshared(seams()), &|| true, &|_| {});
    let got = run(Input::Joint { left: &l, right: &r },
                  Seams { pass_event: Some(&ev), share_geometry: Some(small_batches), bar_progress: Some(&part), ..seams() },
                  &|| true, &done);
    same(&got, &want, "mixed");
    let log = log.into_inner().unwrap();
    let built: Vec<usize> = log.iter().filter_map(|e| match e { Ev::P(PassEvent::Built(n)) if *n > 0 => Some(*n), _ => None }).collect();
    assert!(!built.is_empty(), "setup: no shared-block pass ran");
    assert!(got.work.shared_members > 0, "setup: the no-blocks pass did not share");
    // Exactly once per bar.
    let mut seen = vec![0usize; BARS];
    for e in &log { if let Ev::Done(b) = e { seen[*b] += 1; } }
    assert!(seen.iter().all(|&c| c == 1), "callbacks per bar: {seen:?}");
    // Never early. A bar whose last pass is a shared-block pass: after its
    // `Bar(that pass, bar)`. A bar finished by the no-blocks pass: after its
    // batch's last member index is folded.
    let batch_of = |b: usize| -> Option<(usize, usize)> {
        // The batch containing `b`: the run of `Bar(0, _)` events it is in.
        let mut first = None;
        for e in &log {
            match e {
                Ev::P(PassEvent::Bar(0, x)) => {
                    if first.is_none() { first = Some(*x); }
                    if *x == b { return first.map(|f| (f, 0)); }
                }
                _ => first = None,
            }
        }
        None
    };
    // Partial progress: only for bars not yet done, below 1, never decreasing.
    let mut last = [0.0f32; BARS];
    let mut parts = 0;
    for (pos, e) in log.iter().enumerate() {
        let Ev::Part(b, f) = e else { continue };
        parts += 1;
        assert!(*f > 0.0 && *f < 1.0, "bar {b}: partial progress {f}");
        assert!(*f >= last[*b], "bar {b}: progress went backwards");
        last[*b] = *f;
        assert!(!log[..pos].iter().any(|e| matches!(e, Ev::Done(x) if x == b)), "bar {b}: progress after done");
    }
    assert!(parts > 0, "setup: no partial progress reported");
    for (pos, e) in log.iter().enumerate() {
        let Ev::Done(b) = e else { continue };
        let before = &log[..pos];
        let later_pass = before.iter().rev().find_map(|e| match e { Ev::P(PassEvent::Bar(n, x)) if *x == *b && *n > 0 => Some(*n), _ => None });
        if later_pass.is_some() { continue; }
        if let Some((first, _)) = batch_of(*b) {
            assert!(before.iter().any(|e| matches!(e, Ev::P(PassEvent::SharedIndexDone { first_bar, member: 8 }) if *first_bar == first)),
                    "bar {b} reported done before its batch's last member index");
            // And no shared-block pass still to come for it.
            assert!(!log[pos..].iter().any(|e| matches!(e, Ev::P(PassEvent::Bar(n, x)) if *x == *b && *n > 0)),
                    "bar {b} reported done before a later pass ran its members");
        }
    }
}

/// Cancellation raised after a known shared window: no further window or
/// subbatch of the shared pass starts, nothing is published, and every
/// reservation is released — through the configured analysis pool. Its
/// groups are routed by `never_gpu`, which parks no calibration decision, so
/// the empty-slot check below is a guard and not evidence of retirement;
/// `cancelling_a_shared_window_retires_a_parked_calibration_decision` is.
#[test]
fn cancellation_after_a_shared_window_stops_the_pass_and_publishes_nothing() {
    use super::super::gpu_calib as calib;
    let (_, l, r) = signals(2.5).swap_remove(0);
    let _env = calib::aslt_test_env("shared_pass_cancel");
    let probe = ShareBudget::new(SHARE_GEOMETRY.budget_bytes);
    let stop = AtomicBool::new(false);
    let windows_after = AtomicUsize::new(0);
    let prepared_after = AtomicUsize::new(0);
    let windows = AtomicUsize::new(0);
    let hook = |e: PassEvent| match e {
        PassEvent::SharedWindow { .. } => {
            if stop.load(Ordering::SeqCst) { windows_after.fetch_add(1, Ordering::SeqCst); }
            // Cancel at a known boundary: after the second window has completed.
            if windows.fetch_add(1, Ordering::SeqCst) == 1 { stop.store(true, Ordering::SeqCst); }
        }
        PassEvent::SharedPrepared { .. } if stop.load(Ordering::SeqCst) => {
            prepared_after.fetch_add(1, Ordering::SeqCst);
        }
        _ => {}
    };
    let done = AtomicUsize::new(0);
    let (out, clean) = calib::install(|| {
        let out = analyze_routed(
            Input::Joint { left: &l, right: &r }, SR, BARS, 20.0, 22_050.0, hop_for_fps(SR, 180.0), &cfg(),
            &|| !stop.load(Ordering::SeqCst), &|_| { done.fetch_add(1, Ordering::SeqCst); }, never_gpu,
            Seams { shared_budget: 0, share_probe: Some(&probe), pass_event: Some(&hook), ..Seams::production() },
        );
        (out, calib::pending_is_empty())
    });
    assert!(windows.load(Ordering::SeqCst) >= 2, "setup: the shared pass never reached its second window");
    assert_eq!(windows_after.load(Ordering::SeqCst), 0, "a shared window ran after cancellation");
    assert_eq!(prepared_after.load(Ordering::SeqCst), 0, "a shared subbatch started after cancellation");
    assert!(out.mix.is_empty() && out.left.is_empty() && out.right.is_empty() && out.codes.is_none(),
            "a cancelled analysis published data");
    assert!(probe.peak() > 0, "setup: nothing was reserved");
    assert_eq!(probe.used(), 0, "a reservation outlived the cancelled pass");
    assert!(clean, "a cancelled analysis left something parked on the pool thread");
    assert!(done.load(Ordering::SeqCst) < BARS, "every bar was reported done on a cancelled run");
}

/// The calibration half of cancellation, with its premise established rather
/// than assumed: `never_gpu` above parks no decision, so the check there cannot
/// fail. Here an eligible group is routed through the calibration's Auto
/// decision, which explores the cores first on a fresh identity (an injected
/// adapter and isolated directory, no device). The decision is witnessed
/// parked — for that group, routed to the cores, reserved and started — on the
/// thread that owns the group; cancellation is raised after one of that
/// group's shared windows completes; and after the group unwinds the same
/// thread holds nothing. The started attempt is not refunded (that is the
/// policy), but it is no longer outstanding, and no sample landed against it.
/// A following analysis on the same thread, whose groups take no decision of
/// their own, records nothing: it did not inherit the cancelled group's.
#[test]
fn cancelling_a_shared_window_retires_a_parked_calibration_decision() {
    use super::super::gpu_calib as calib;
    use std::thread::{self, ThreadId};
    type Witness = (ThreadId, Option<(usize, bool, bool, bool)>);
    static ROUTED: Mutex<Vec<(usize, Witness)>> = Mutex::new(Vec::new());
    fn via_calibration(n: usize) -> bool {
        let r = calib::use_device(n);
        ROUTED.lock().unwrap().push((n, (thread::current().id(), calib::pending_for_test())));
        r
    }
    // 4 s at 48 kHz, 256 bars, 0.9 s windows: groups from 2^13 up to the
    // routing floor, 2^17, which is the first (largest) group.
    let (sr, bars, lo, hi) = (48_000u32, 256usize, 40.0f32, 18_000.0f32);
    let len = (sr as f32 * 4.0) as usize;
    let l: Vec<f32> = (0..len).map(|i| (i as f32 * 0.0023).sin() * 0.7 + (i as f32 * 0.019).sin() * 0.3).collect();
    let r: Vec<f32> = (0..len).map(|i| (i as f32 * 0.0031 + 0.4).sin() * 0.5 + (i as f32 * 0.011).sin() * 0.2).collect();
    let cfg = AsltConfig { max_window_s: 0.9, ..AsltPreset::Fast.config() };
    let hop = hop_for_fps(sr, 120.0);
    let n = GPU_MIN_BLOCK;

    let _env = calib::aslt_test_env("shared_pass_decision_cancel");
    assert!(_env.dir().starts_with(std::env::temp_dir()), "setup: calibration not isolated");
    let recorded = || std::mem::take(&mut *calib::RECORDED.lock().unwrap_or_else(|e| e.into_inner()));
    recorded();
    ROUTED.lock().unwrap().clear();

    let stop = AtomicBool::new(false);
    let started: Mutex<Vec<Witness>> = Mutex::new(Vec::new());
    let group = AtomicUsize::new(0);
    let (windows, windows_after, prepared_after) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
    let hook = |e: PassEvent| match e {
        // Emitted by the group's own thread as its work begins.
        PassEvent::WorkStarted => {
            let w = calib::pending_for_test();
            group.store(w.map_or(0, |d| d.0), Ordering::SeqCst);
            started.lock().unwrap().push((thread::current().id(), w));
        }
        PassEvent::SharedWindow { .. } if group.load(Ordering::SeqCst) == n => {
            if stop.load(Ordering::SeqCst) { windows_after.fetch_add(1, Ordering::SeqCst); }
            if windows.fetch_add(1, Ordering::SeqCst) == 0 { stop.store(true, Ordering::SeqCst); }
        }
        PassEvent::SharedPrepared { .. } if stop.load(Ordering::SeqCst) => {
            prepared_after.fetch_add(1, Ordering::SeqCst);
        }
        _ => {}
    };
    let probe = ShareBudget::new(SHARE_GEOMETRY.budget_bytes);
    let input = Input::Joint { left: &l, right: &r };
    let (out, owner, after, attempts, samples, later, later_after, later_samples) = calib::install(|| {
        let out = analyze_routed(
            input, sr, bars, lo, hi, hop, &cfg, &|| !stop.load(Ordering::SeqCst), &|_| {}, via_calibration,
            Seams { shared_budget: 0, share_probe: Some(&probe), pass_event: Some(&hook), ..Seams::production() },
        );
        let after = calib::pending_for_test();
        let (attempts, samples) = (calib::attempts_for_test(n), calib::samples_for_test(n));
        // The next analysis on this thread: its groups take no decision, so
        // anything they record could only be one left behind.
        let short = &l[..sr as usize];
        let later = analyze_routed(Input::Mono(short), sr, bars, lo, hi, hop, &cfg, &|| true, &|_| {}, never_gpu,
                                   Seams { shared_budget: 0, ..Seams::production() });
        (out, thread::current().id(), after, attempts, samples, later, calib::pending_for_test(), calib::samples_for_test(n))
    });

    // Setup: the decision existed, on the owning thread, for the eligible
    // group, routed to the cores, exploratory, and its work had begun.
    let routed = std::mem::take(&mut *ROUTED.lock().unwrap());
    let (rn, (rt, rd)) = *routed.first().expect("setup: no group was routed");
    assert_eq!(rn, n, "setup: the first group is 2^{}, not the routing floor", rn.trailing_zeros());
    assert_eq!(rt, owner, "setup: the group was routed on another thread than the one that owns the analysis");
    assert_eq!(rd, Some((n, false, true, false)), "setup: no exploratory cores decision parked for 2^17");
    let started = started.into_inner().unwrap();
    assert_eq!(started.first(), Some(&(owner, Some((n, false, true, true)))),
               "setup: the decision was not parked, started, on the owning thread when the group's work began");
    assert!(windows.load(Ordering::SeqCst) >= 1, "setup: the eligible group never completed a shared window");
    assert!(probe.peak() > 0, "setup: the eligible group reserved nothing for sharing");

    // The cancellation stopped the shared pass and published nothing.
    assert_eq!(windows_after.load(Ordering::SeqCst), 0, "a shared window ran after cancellation");
    assert_eq!(prepared_after.load(Ordering::SeqCst), 0, "a shared subbatch started after cancellation");
    assert!(out.mix.is_empty() && out.codes.is_none(), "a cancelled analysis published data");
    assert_eq!(probe.used(), 0, "a sharing reservation outlived the cancelled pass");
    assert_eq!(routed.len(), 1, "a group after the cancelled one was routed: {routed:?}");

    // The decision is retired on its own thread; its started attempt is
    // accounted for, not left outstanding; it landed no sample.
    assert_eq!(after, None, "the cancelled group's decision is still parked on its thread");
    // Fresh identity, one exploratory decision: one attempt, started, so it
    // stays charged (no refund once work began) and is no longer outstanding.
    assert_eq!(attempts, Some((1, 0)),
               "the cancelled group's started attempt was not kept charged and retired (attempts, outstanding)");
    assert_eq!(samples, Some((0, 0)), "the cancelled group landed a sample");

    // The next analysis inherited nothing and recorded nothing against it.
    assert!(!later.mix.is_empty(), "setup: the following analysis produced nothing");
    assert!(later.work.groups_size_sequential > 0, "setup: the following analysis ran no cores group");
    assert_eq!(later_after, None, "the following analysis left a decision parked");
    assert_eq!(later_samples, Some((0, 0)), "a later group recorded a sample against the cancelled decision");
    assert!(recorded().is_empty(), "a decision was recorded after the cancellation");
}

/// Ineligible work is untouched: with the block budget admitting every size
/// (no no-blocks pass), and on the device route, sharing does nothing.
#[test]
fn sharing_leaves_the_shared_block_and_device_routes_alone() {
    let (_, l, r) = signals(2.5).swap_remove(0);
    let roomy = || Seams { shared_budget: usize::MAX / 4, ..Seams::production() };
    let got = run(Input::Joint { left: &l, right: &r }, roomy(), &|| true, &|_| {});
    let want = run(Input::Joint { left: &l, right: &r }, unshared(roomy()), &|| true, &|_| {});
    same(&got, &want, "every size shared");
    assert_eq!(got.work.shared_members, 0, "the shared-block route used the no-blocks sharing");
    assert!(got.work.groups_size_sequential > 0);
    // A group routed to the device does not run size-sequential at all.
    fn all_device(_: usize) -> bool { true }
    let dev = analyze_routed(Input::Joint { left: &l, right: &r }, SR, BARS, 20.0, 22_050.0, hop_for_fps(SR, 180.0),
                             &cfg(), &|| true, &|_| {}, all_device, Seams { shared_budget: 0, ..Seams::production() });
    assert_eq!(dev.work.shared_members, 0, "a device-routed group shared on the cores");
}

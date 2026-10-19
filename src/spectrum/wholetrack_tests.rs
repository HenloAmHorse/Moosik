//! The whole-track view's contract, geometry and raster, against direct
//! references. Every expected picture here is computed by scanning the plane's
//! codes cell by cell, never by the code under test.
use super::*;
use crate::spectrum::{
    aslt, freq_scale::FreqScale, settings_for, BarMappingMode, InterpolationMode, WindowFn,
};
use std::path::Path;

const BARS: usize = 64;
const FRAMES: usize = 1000;

/// A fixture whose every interesting cell is known:
/// * tones on bars 10 and 40 for the whole track;
/// * a chirp rising one bar per 20 frames;
/// * one-frame transients at frames 0, 500 and 999, all bars;
/// * frames 300..320 silent (code 0) and frames 600..605 clipped (max code).
fn fixture() -> PreFrames {
    let mut codes = vec![0u16; FRAMES * BARS];
    for f in 0..FRAMES {
        let row = &mut codes[f * BARS..(f + 1) * BARS];
        for (b, c) in row.iter_mut().enumerate() {
            *c = ((f * 7 + b * 13) % 200) as u16 + 100; // a low, varied floor
        }
        row[10] = 3000;
        row[40] = 2500;
        row[(f / 20) % BARS] = 3500;
        if [0, 500, 999].contains(&f) {
            row.iter_mut().for_each(|c| *c = 4000);
        }
        if (300..320).contains(&f) {
            row.iter_mut().for_each(|c| *c = 0);
        }
        if (600..605).contains(&f) {
            row.iter_mut().for_each(|c| *c = 4095);
        }
    }
    PreFrames::from_v4_codes(codes, FRAMES, BARS)
}

/// Direct reference: every pixel's block, scanned cell by cell.
fn reference(p: &PreFrames, v: View, w: usize, h: usize, data_bars: usize) -> Vec<u32> {
    let mut out = vec![OUTSIDE; w * h];
    for y in 0..h {
        let (g0, g1) = span(v.b0, v.b1, h, h - 1 - y, p.bars());
        for x in 0..w {
            let (f0, f1) = span(v.t0, v.t1, w, x, p.len());
            if f1 <= f0 || g1 <= g0 {
                continue;
            }
            if g0 >= data_bars {
                out[y * w + x] = NO_DATA;
                continue;
            }
            let mut m = 0u16;
            for f in f0..f1 {
                for g in g0..g1.min(data_bars) {
                    m = m.max(p.row(f).unwrap()[g]);
                }
            }
            out[y * w + x] = m as u32;
        }
    }
    out
}

fn raster(p: &PreFrames, v: View, w: usize, h: usize, data_bars: usize) -> Raster {
    rasterize(p, v, w, h, data_bars, &AtomicBool::new(false)).unwrap()
}

#[test]
fn every_view_equals_a_direct_scan_of_the_codes() {
    let p = fixture();
    let views = [
        View::full(FRAMES, BARS),
        View { t0: 0.0, t1: 37.0, b0: 0.0, b1: 9.0 },            // magnified, at the start
        View { t0: 963.0, t1: 1000.0, b0: 55.0, b1: 64.0 },      // magnified, at the end
        View { t0: 123.4, t1: 777.7, b0: 3.3, b1: 51.9 },        // fractional, off-centre
        View { t0: -50.0, t1: 1100.0, b0: 0.0, b1: 64.0 },       // past both ends
        View { t0: 499.5, t1: 500.5, b0: 9.5, b1: 10.5 },        // one cell
    ];
    for v in views {
        for (w, h) in [(1, 1), (7, 3), (100, 64), (333, 17), (1000, 64), (2048, 200)] {
            for data_bars in [BARS, 50] {
                let r = raster(&p, v, w, h, data_bars);
                assert_eq!(r.codes, reference(&p, v, w, h, data_bars), "view {v:?} at {w}x{h}, data_bars {data_bars}");
            }
        }
    }
}

#[test]
fn a_one_frame_transient_survives_any_reduction() {
    let p = fixture();
    for w in [1, 2, 3, 10, 99, 250, 999] {
        let r = raster(&p, View::full(FRAMES, BARS), w, 8, BARS);
        for f in [0usize, 500, 999] {
            // The column holding frame f must show the transient's code on
            // every row.
            let x = (0..w).find(|&x| { let (a, b) = span(0.0, FRAMES as f64, w, x, FRAMES); (a..b).contains(&f) }).unwrap();
            for y in 0..8 {
                assert!(r.codes[y * w + x] >= 4000, "frame {f} lost at width {w}, row {y}");
            }
        }
    }
}

#[test]
fn reduced_spans_tile_the_axis_and_magnified_ones_are_single_cells() {
    for (a0, a1, n, limit) in [(0.0, 1000.0, 7, 1000), (12.5, 987.25, 333, 1000), (0.0, 64.0, 10, 64)] {
        let s = spans(a0, a1, n, limit);
        // Contiguous cover with at most a shared boundary cell.
        assert_eq!(s[0].0, a0.floor() as usize);
        assert_eq!(s[n - 1].1, (a1.ceil() as usize).min(limit));
        for w in s.windows(2) {
            assert!(w[1].0 <= w[0].1 && w[1].0 + 1 >= w[0].1, "gap or overlap between {:?} and {:?}", w[0], w[1]);
        }
    }
    for (x, &(lo, hi)) in spans(100.0, 110.0, 40, 1000).iter().enumerate() {
        assert_eq!(hi - lo, 1);
        assert_eq!(lo, 100 + x / 4, "pixel {x} is not the cell under its centre");
    }
}

#[test]
fn the_top_row_is_the_highest_bar_and_time_runs_left_to_right() {
    let mut codes = vec![0u16; 4 * 4];
    codes[3 * 4] = 77; // frame 3, bar 0
    codes[3] = 99; // frame 0, bar 3
    let p = PreFrames::from_v4_codes(codes, 4, 4);
    let r = raster(&p, View::full(4, 4), 4, 4, 4);
    assert_eq!(r.codes[0], 99, "bar 3 of frame 0 belongs top-left");
    assert_eq!(r.codes[3 * 4 + 3], 77, "bar 0 of frame 3 belongs bottom-right");
    assert!(r.native());
}

#[test]
fn floor_clipped_and_no_data_stay_distinct() {
    let p = fixture();
    let r = raster(&p, View { t0: 300.0, t1: 320.0, b0: 0.0, b1: 64.0 }, 20, 64, 50);
    assert!(r.codes[63 * 20..].iter().all(|&c| c == 0), "silence is the floor code");
    assert!(r.codes[..14 * 20].iter().all(|&c| c == NO_DATA), "bars 50.. are no data, not floor");
    let r = raster(&p, View { t0: 600.0, t1: 605.0, b0: 0.0, b1: 64.0 }, 5, 64, BARS);
    assert!(r.codes.iter().all(|&c| c == 4095));
    let r = raster(&p, View { t0: 990.0, t1: 1010.0, b0: 0.0, b1: 64.0 }, 20, 4, BARS);
    assert_eq!(r.codes[19], OUTSIDE, "past the last frame is outside, not floor");
}

#[test]
fn cancellation_returns_nothing() {
    assert!(rasterize(&fixture(), View::full(FRAMES, BARS), 100, 10, BARS, &AtomicBool::new(true)).is_none());
}

#[test]
fn the_raster_budget_holds_for_any_window() {
    for (w, h, planes, side) in [(10_000.0, 10_000.0, 1, 16_384), (3840.0, 2160.0, 2, 8192), (5000.0, 300.0, 1, 2048), (1.0, 1.0, 2, 2048)] {
        let (rw, rh) = fit_raster(w, h, planes, side);
        assert!(rw * rh * planes <= MAX_RASTER_PIXELS, "{w}x{h}x{planes} -> {rw}x{rh}");
        assert!(rw <= side && rh <= side && rw >= 1 && rh >= 1);
    }
}

// ── geometry ────────────────────────────────────────────────────────────────

fn record(mapping: BarMappingMode, scale: FreqScale, sr: u32, lo: f32, hi: f32) -> String {
    let mut cfg = aslt::AsltPreset::Standard.config();
    cfg.scale = scale;
    settings_for(
        Path::new("C:/x/track.flac"), 512, 4096, 2, 0.75, &WindowFn::Hann, lo, hi, &mapping,
        &InterpolationMode::None, 2_822_400, &cfg, 180.0,
    )
    .record(sr)
}

#[test]
fn the_record_states_the_producers_grid() {
    let scales = [
        FreqScale::Log,
        FreqScale::Erb,
        FreqScale::Blend(-0.4),
        FreqScale::Lens { tilt: 0.3, hz: 2000.0, oct: 2.0, gain: 3.0 },
    ];
    for scale in scales {
        for sr in [44_100u32, 96_000] {
            // Superlet: spread to min(hi, Nyquist), centres exactly as aslt.
            let g = Geometry::parse(&record(BarMappingMode::Superlet, scale, sr, 20.0, 24_000.0)).unwrap();
            let a_max = (sr as f32 / 2.0).min(24_000.0);
            assert_eq!((g.bars, g.sample_rate, g.scale, g.transform), (512, sr, scale, Transform::Superlet));
            assert_eq!(g.hop, aslt::hop_for_fps(sr, 180.0));
            assert_eq!(g.axis_hi, a_max);
            assert_eq!(g.data_bars, 512);
            for i in [0, 1, 255, 511] {
                assert_eq!(g.bar_center(i), scale.bar_center(i, 512, 20.0, a_max));
            }
            // FFT: spread to hi whatever Nyquist is, and bars wholly above it
            // are no data.
            let g = Geometry::parse(&record(BarMappingMode::Cqt, scale, sr, 20.0, 24_000.0)).unwrap();
            assert_eq!(g.transform, Transform::Fft { size: 4096, pad: 2, mapping: Mapping::CqWeighted });
            assert_eq!(g.hop, 1024);
            assert_eq!(g.axis_hi, 24_000.0);
            let nyq = sr as f32 / 2.0;
            for i in 0..512 {
                let (lo, _) = g.bar_edges(i);
                assert_eq!(i < g.data_bars, lo < nyq, "bar {i} at {lo} Hz, Nyquist {nyq}");
            }
            if sr == 44_100 {
                assert!(g.data_bars < 512, "24 kHz is above a 44.1 kHz Nyquist");
            } else {
                assert_eq!(g.data_bars, 512);
            }
        }
    }
}

#[test]
fn records_without_geometry_are_refused() {
    let good = record(BarMappingMode::Superlet, FreqScale::Log, 44_100, 20.0, 20_000.0);
    assert!(Geometry::parse(&good).is_ok());
    for bad in [
        good.replace(";sr=44100", ""),
        good.replace(";hop=", ";hop=x"),
        good.replace("scale=log", "scale=cqt"),
        good.replace("moosik-analysis/1", "moosik-analysis/0"),
        good.replace(";bars=512", ";bars=0"),
        String::new(),
    ] {
        assert!(Geometry::parse(&bad).is_err(), "accepted {bad}");
    }
}

#[test]
fn frame_times_follow_the_stored_hop() {
    let g = Geometry::parse(&record(BarMappingMode::Superlet, FreqScale::Log, 48_000, 20.0, 20_000.0)).unwrap();
    let hop = aslt::hop_for_fps(48_000, 180.0) as f64;
    assert_eq!(g.frame_time(1000.0), 1000.0 * hop / 48_000.0);
    assert!(g.transform.anchor_note(1.0, 48_000).contains("centred"));
    let g = Geometry::parse(&record(BarMappingMode::FlatOverlap, FreqScale::Log, 48_000, 20.0, 20_000.0)).unwrap();
    assert_eq!(g.frame_time(10.0), 10.0 * 1024.0 / 48_000.0);
    assert!(g.transform.anchor_note(1.0, 48_000).contains("starts at the frame time"));
}

// ── zoom ────────────────────────────────────────────────────────────────────

#[test]
fn zoom_keeps_the_point_under_the_pointer() {
    use crate::spectrum::wholetrack_ui::zoom_view;
    let v = View { t0: 200.0, t1: 600.0, b0: 10.0, b1: 50.0 };
    for (at, kt, kf) in [((250.0, 12.0), 0.5, 1.0), ((590.0, 49.0), 0.25, 0.5), ((400.0, 30.0), 1.5, 1.2)] {
        let z = zoom_view(v, at, kt, kf, FRAMES as f64, BARS as f64);
        let frac = |a: f64, lo: f64, hi: f64| (a - lo) / (hi - lo);
        assert!((frac(at.0, v.t0, v.t1) - frac(at.0, z.t0, z.t1)).abs() < 1e-9, "time anchor moved: {z:?}");
        assert!((frac(at.1, v.b0, v.b1) - frac(at.1, z.b0, z.b1)).abs() < 1e-9, "bar anchor moved: {z:?}");
        assert!(((z.t1 - z.t0) - (v.t1 - v.t0) * kt).abs() < 1e-9);
    }
    // Time and frequency are independent.
    let z = zoom_view(v, (300.0, 20.0), 0.5, 1.0, FRAMES as f64, BARS as f64);
    assert_eq!((z.b0, z.b1), (v.b0, v.b1));
    // At an edge the view stops at the track, not past it, and never below
    // the smallest span.
    let z = zoom_view(v, (210.0, 11.0), 100.0, 100.0, FRAMES as f64, BARS as f64);
    assert_eq!(z, View::full(FRAMES, BARS));
    let z = zoom_view(v, (300.0, 30.0), 1e-9, 1e-9, FRAMES as f64, BARS as f64);
    assert!((z.t1 - z.t0 - 8.0).abs() < 1e-9 && (z.b1 - z.b0 - 4.0).abs() < 1e-9);
}

// ── the worker ──────────────────────────────────────────────────────────────

fn job(epoch: u64, seq: u64, p: &Arc<PreFrames>) -> Job {
    Job { epoch, seq, planes: vec![p.clone()], data_bars: BARS, overview: None, cols: Vec::new(), accel: Vec::new(), window: (0, 0) }
}

fn wait_for(w: &mut Worker, epoch: u64) -> Vec<Done> {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let d = w.poll(epoch);
        if !d.is_empty() || !w.busy() {
            return d;
        }
        assert!(Instant::now() < deadline, "the worker never finished");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn requests_coalesce_into_one_running_and_one_waiting() {
    let p = Arc::new(fixture());
    let mut w = Worker::default();
    let gate = Arc::new(AtomicBool::new(true));
    w.hold = Some(gate.clone());
    for seq in 1..=50 {
        w.request(job(1, seq, &p));
    }
    assert_eq!(w.started, 1, "a burst started more than one thread");
    gate.store(false, Ordering::SeqCst);
    let mut seen = Vec::new();
    while w.busy() {
        seen.extend(wait_for(&mut w, 1).into_iter().map(|d| d.seq));
    }
    assert_eq!(seen, vec![1, 50], "only the first and the newest request may run");
    assert_eq!(w.started, 2);
    assert_eq!(w.peak_live, 1);
}

#[test]
fn a_result_for_an_abandoned_epoch_is_never_installed() {
    let p = Arc::new(fixture());
    let mut w = Worker::default();
    let gate = Arc::new(AtomicBool::new(true));
    w.hold = Some(gate.clone());
    w.request(job(1, 1, &p));
    // Finished rasterising and about to send when the view moves on (track,
    // plane or settings changed): only the epoch check can refuse it now.
    while !w.parked.load(Ordering::SeqCst) {
        std::thread::yield_now();
    }
    w.invalidate();
    gate.store(false, Ordering::SeqCst);
    let deadline = Instant::now() + Duration::from_secs(20);
    while w.busy() {
        assert!(w.poll(2).is_empty(), "a late result reached the new epoch");
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    // Everything the thread held is released once it has exited.
    let deadline = Instant::now() + Duration::from_secs(20);
    while Arc::strong_count(&p) > 1 {
        assert!(Instant::now() < deadline, "the worker kept the plane alive");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn a_new_epoch_cancels_the_running_job_and_drops_the_waiting_one() {
    let p = Arc::new(fixture());
    let mut w = Worker::default();
    let gate = Arc::new(AtomicBool::new(true));
    w.hold = Some(gate.clone());
    w.request(job(1, 1, &p));
    w.request(job(1, 2, &p));
    w.request(job(2, 3, &p)); // replaces the waiting epoch-1 job
    gate.store(false, Ordering::SeqCst);
    let mut seen = Vec::new();
    while w.busy() {
        seen.extend(wait_for(&mut w, 2).into_iter().map(|d| (d.epoch, d.seq)));
    }
    assert_eq!(seen, vec![(2, 3)]);
    assert!(w.discarded <= 1);
}

#[test]
fn the_overview_is_drawn_from_its_own_plane_and_survives_coalescing() {
    let p = Arc::new(fixture());
    let other = Arc::new(PreFrames::from_v4_codes(vec![9; FRAMES * BARS], FRAMES, BARS));
    let mut w = Worker::default();
    let gate = Arc::new(AtomicBool::new(true));
    w.hold = Some(gate.clone());
    w.request(job(1, 1, &other));
    w.request(Job { overview: Some((p.clone(), (100, 8))), ..job(1, 2, &other) });
    w.request(job(1, 3, &other)); // must not lose the strip
    gate.store(false, Ordering::SeqCst);
    let mut ov = None;
    while w.busy() {
        for d in wait_for(&mut w, 1) {
            if d.overview.is_some() {
                ov = d.overview;
            }
        }
    }
    let ov = ov.expect("the overview request was lost");
    assert_eq!(ov.codes, reference(&p, View::full(FRAMES, BARS), 100, 8, BARS));
}

// ── hover ───────────────────────────────────────────────────────────────────

fn snapshot_of(p: PreFrames, rec: &str) -> Snapshot {
    let geometry = Geometry::parse(rec).unwrap();
    let mix = Arc::new(p);
    Snapshot { revision: 0, origin: Origin::Memory, pin: None, inv_max: mix.inv_max(), max_code: mix.max_code(), lr: None, mix, geometry }
}

#[test]
fn hover_reads_the_stored_cell_and_says_what_it_is() {
    use crate::spectrum::wholetrack_ui::hover;
    let rec = record(BarMappingMode::Cqt, FreqScale::Log, 44_100, 20.0, 24_000.0)
        .replace("bars=512", &format!("bars={BARS}"));
    let s = snapshot_of(fixture(), &rec);
    let g = s.geometry;
    // Native: a 10-frame × 64-bar window at 10 × 64 pixels.
    let v = View { t0: 495.0, t1: 505.0, b0: 0.0, b1: 64.0 };
    let txt = hover(&s.mix, &v, (10, 64), (5, 63 - 10), &s, "Mix");
    assert!(txt.contains("frame 500") && txt.contains("code 4000/4095"), "{txt}");
    let txt = hover(&s.mix, &v, (10, 64), (4, 63 - 10), &s, "Mix");
    assert!(txt.contains("frame 499") && txt.contains("bar 10") && txt.contains("code 3000/4095"), "{txt}");
    let db = code_db(3000, s.inv_max);
    assert!(txt.contains(&format!("{db:.1} dB")), "{txt}");
    // Floor and top are named as clipped.
    let v = View { t0: 300.0, t1: 310.0, b0: 0.0, b1: 64.0 };
    assert!(hover(&s.mix, &v, (10, 64), (0, 0), &s, "Mix").contains("floor, clipped"));
    let v = View { t0: 600.0, t1: 605.0, b0: 0.0, b1: 64.0 };
    assert!(hover(&s.mix, &v, (5, 64), (0, 63), &s, "Mix").contains("top, clipped"));
    // Pooled: the block and the position of its maximum.
    let txt = hover(&s.mix, &View::full(FRAMES, BARS), (10, 8), (5, 7), &s, "Mix");
    assert!(txt.contains("max of") && txt.contains("at frame 500"), "{txt}");
    // Above Nyquist: no data, not a level.
    if g.data_bars < BARS {
        let txt = hover(&s.mix, &View::full(FRAMES, BARS), (10, 64), (3, 0), &s, "Mix");
        assert!(txt.contains("no analysed data"), "{txt}");
    }
}

/// The schedule Codex found: the worker sends its result after `poll` has
/// drained an empty queue and exits before `poll` reaps the slot. The slot
/// must not be reaped on an exit observed after the drain, or the result
/// stays queued with `busy()` false and nothing polls it again.
#[test]
fn a_result_sent_between_the_drain_and_the_reap_is_never_stranded() {
    let p = Arc::new(fixture());
    let mut w = Worker::default();
    let gate = Arc::new(AtomicBool::new(true));
    w.hold = Some(gate.clone());
    let live = w.live.clone();
    w.request(job(1, 1, &p));
    while !w.parked.load(Ordering::SeqCst) {
        std::thread::yield_now();
    }
    let g = gate.clone();
    w.between_drain_and_reap = Some(Box::new(move || {
        // Publication and exit, exactly inside the window.
        g.store(false, Ordering::SeqCst);
        while live.load(Ordering::SeqCst) != 0 {
            std::thread::yield_now();
        }
    }));
    assert!(w.poll(2).is_empty(), "the drain ran before the send");
    w.between_drain_and_reap = None;
    // Either the slot is still held (so it will be polled again), or nothing
    // is left queued. Never both "idle" and "a result waiting".
    if !w.busy() {
        assert!(w.take_queued().is_none(), "a result was stranded in the queue of an idle worker");
    }
    // The next poll (the retired view polls every tick) disposes of it.
    assert!(w.poll(2).is_empty(), "a result for an abandoned epoch was returned");
    assert!(!w.busy());
    assert!(w.take_queued().is_none(), "the abandoned result is still queued");
    assert_eq!(w.discarded, 1, "the abandoned result was not taken and dropped");
    assert_eq!(Arc::strong_count(&p), 1, "the plane is still held");
}

/// Frequency labels never crowd a short plot, and a deep frequency zoom with
/// no standard tick inside still names its top and bottom frequencies.
#[test]
fn frequency_labels_fit_small_and_deep_zoomed_plots() {
    use crate::spectrum::wholetrack_ui::freq_ticks;
    let g = Geometry::parse(&record(BarMappingMode::Superlet, FreqScale::Log, 48_000, 20.0, 20_000.0)).unwrap();
    let full = View::full(1000, g.bars);
    for h in [60.0f32, 120.0, 400.0] {
        let t = freq_ticks(h, &full, &g);
        assert!(t.len() >= 2, "height {h}: {t:?}");
        for w in t.windows(2) {
            assert!((w[0].1 - w[1].1).abs() >= 13.0, "height {h}: labels {:?} and {:?} overlap", w[0], w[1]);
        }
    }
    // Between two standard ticks (1k and 2k on a log axis): edge labels.
    let (b0, b1) = (g.bar_pos(1100.0), g.bar_pos(1300.0));
    let deep = View { t0: 0.0, t1: 1000.0, b0, b1 };
    let t = freq_ticks(300.0, &deep, &g);
    assert_eq!(t.len(), 2, "{t:?}");
    assert!((t[0].0 - 1300.0).abs() < 5.0 && (t[1].0 - 1100.0).abs() < 5.0, "{t:?}");
    assert_eq!(t[0].2, "1.30k");
}

/// A worker job for a few leftover columns builds its level over the asking
/// camera's frames, so the next zoom step of that camera (all of whose
/// columns change span) is answered by it on the UI thread; built over the
/// leftover columns alone, it answered almost none of them, and a run of
/// wheel steps kept going back to the worker (the 1.5.4 owner report).
#[test]
fn a_level_built_for_leftover_columns_covers_the_camera() {
    let p = fixture();
    let no = std::sync::atomic::AtomicBool::new(false);
    let cam = (100, 900);
    // Two leftover columns of six frames, at the camera's middle.
    let cols = [(0, (500, 506)), (1, (506, 512))];
    let a = Accel::default();
    let out = columns(&p, Some(&a), &cols, cam, BARS, &no).unwrap();
    let l = a.level.lock().unwrap().clone().expect("no level built");
    assert_eq!(l.k, 2);
    // The next zoom step: six-frame columns across the whole camera.
    for f in (cam.0..cam.1 - 6).step_by(6) {
        assert!(l.answers((f, f + 6)), "column {f}..{} is not answered by the level {}..{}", f + 6, l.start, l.end);
    }
    // Same maxima as a direct scan.
    let mut want = vec![0u16; BARS];
    for (k, &(_, s)) in cols.iter().enumerate() {
        column_max(&p, s, &mut want);
        assert_eq!(&out[k * BARS..(k + 1) * BARS], &want[..]);
    }
    // Without the camera, only the leftover columns' neighbourhood.
    let b = Accel::default();
    columns(&p, Some(&b), &cols, (0, 0), BARS, &no).unwrap();
    let lb = b.level.lock().unwrap().clone().unwrap();
    assert!(!lb.answers((cam.0, cam.0 + 6)), "setup: without the camera the level already covered it");
}

/// One installed sliding level can answer a third block without a new
/// octave: all the blocks read must lie inside the requested interval, and
/// together they must cover its middle as well as both ends. Its read count
/// is also the one the UI's admission calculation charges.
#[test]
fn one_two_and_three_level_blocks_equal_direct_maxima_and_charge_their_reads() {
    let p = fixture();
    let no = AtomicBool::new(false);
    // k=0/1 cover the short windows; k=2 covers the 8/9-frame boundary
    // crossed by the owner's wheel burst.
    for k in [0u32, 1, 2] {
        let b = 1usize << k;
        let l = Level::build(&p, BARS, k, 10, 300, &no).unwrap();
        for len in b..=3 * b {
            for f0 in [10, 11, 37, 101, 300 + b - len - 1] {
                let sp = (f0, f0 + len);
                assert!(l.answers(sp), "k {k}, span {sp:?} was not answered");
                let reads = len.div_ceil(b);
                assert_eq!(l.rows_for(sp), reads, "k {k}, span {sp:?}: level reads");
                assert_eq!(held_rows(Some(&l), None, sp), reads, "k {k}, span {sp:?}: held reads");
                assert_eq!(held_cost(Some(&l), None, sp, BARS), (reads * BARS).div_ceil(4), "k {k}, span {sp:?}: charged reads");
                let (mut want, mut got) = (vec![0; BARS], vec![u16::MAX; BARS]);
                column_max(&p, sp, &mut want);
                l.column_max(sp, &mut got);
                assert_eq!(got, want, "k {k}, span {sp:?}: omitted or extra source cell");
                got.fill(u16::MAX);
                assert_eq!(column_max_held(&p, Some(&l), None, sp, &mut got), reads);
                assert_eq!(got, want, "k {k}, span {sp:?}: production held query");
            }
        }
        assert!(!l.answers((10, 10 + 3 * b + 1)), "k {k}: accepted more than three blocks");
        assert!(!l.answers((9, 9 + b)), "k {k}: read before this level");
        assert!(!l.answers((300, 300 + b)), "k {k}: read after this level");
        if b > 1 {
            assert!(!l.answers((10, 10 + b - 1)), "k {k}: a block would include outside cells");
        }
    }
}

/// Merely widening `answers` while still reading the first and last block
/// loses this deliberately isolated middle peak. Reading overlapping blocks
/// outside the requested interval would instead pick either 4095 peak.
#[test]
fn a_three_block_query_keeps_the_middle_peak_and_excludes_both_outside_peaks() {
    let (frames, bars, f0) = (64usize, 4usize, 8usize);
    for len in [9usize, 10, 11, 12] {
        let f1 = f0 + len;
        let mut codes = vec![100u16; frames * bars];
        for b in 0..bars {
            codes[(f0 + 4) * bars + b] = 2000 + 100 * b as u16;
            codes[(f0 - 1) * bars + b] = 4095;
            codes[f1 * bars + b] = 4095;
        }
        let p = PreFrames::from_v4_codes(codes, frames, bars);
        let l = Level::build(&p, bars, 2, 0, frames, &AtomicBool::new(false)).unwrap();
        let sp = (f0, f1);
        assert!(l.answers(sp), "span {sp:?} was not answered");
        assert_eq!(l.rows_for(sp), 3);
        let mut got = vec![u16::MAX; bars];
        l.column_max(sp, &mut got);
        let want: Vec<u16> = (0..bars).map(|b| 2000 + 100 * b as u16).collect();
        assert_eq!(got, want, "span {sp:?}: lost the middle or included an outside peak");
    }
}

/// A magnified column is one source row, copied exactly into a reused
/// destination. A following empty span must not leave that row behind.
#[test]
fn single_frame_and_empty_queries_overwrite_reused_output() {
    let p = fixture();
    for db in [1usize, 17, BARS] {
        let mut out = vec![u16::MAX; db];
        for f in [0, 301, 500, FRAMES - 1] {
            assert_eq!(column_max_held(&p, None, None, (f, f + 1), &mut out), 1);
            assert_eq!(out, p.row(f).unwrap()[..db], "frame {f}, bars {db}: not a row copy");
            assert_eq!(column_max_held(&p, None, None, (f, f), &mut out), 0);
            assert_eq!(out, vec![0; db], "frame {f}, bars {db}: an empty span kept old data");
        }
    }
}

#[test]
fn two_and_three_frame_queries_get_a_short_exact_level() {
    let p = fixture();
    let a = Accel::default();
    let no = AtomicBool::new(false);
    let cols = [(0, (100, 102)), (1, (102, 105)), (2, (104, 106))];
    let got = columns(&p, Some(&a), &cols, (90, 120), BARS, &no).unwrap();
    let level = a.level.lock().unwrap().clone().expect("short level was not built");
    assert_eq!(level.k, 1);
    for (i, &(_, sp)) in cols.iter().enumerate() {
        assert!(level.answers(sp));
        let mut want = vec![0; BARS];
        column_max(&p, sp, &mut want);
        assert_eq!(&got[i * BARS..(i + 1) * BARS], &want);
    }
}

#[test]
fn pyramid_bit_planning_matches_the_original_level_search() {
    let p = fixture();
    let py = Pyramid::build(&p, BARS, &AtomicBool::new(false)).unwrap();
    for f0 in 0..FRAMES {
        for len in [0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 99, FRAMES] {
            let f1 = (f0 + len).min(FRAMES);
            let old = (0..py.levels.len()).rev()
                .map(|k| (k, 1usize << (PYRAMID_FROM as usize + k)))
                .find(|&(_, size)| f0.is_multiple_of(size) && f0 + size <= f1);
            assert_eq!(py.block_at(f0, f1), old, "{f0}..{f1}");
        }
    }
}

/// Query-only measurements, excluding construction, colour conversion and
/// uploads. No admission weight is changed unless the actual work supports it.
#[test]
#[ignore = "wide-query cost probe: run alone in release"]
fn wide_query_cost_probe() {
    use std::hint::black_box;
    let (frames, bars, width) = (65_536usize, 1024usize, 2428usize);
    let codes = (0..frames * bars).map(|i| ((i * 31 + i / bars * 17) % 4096) as u16).collect();
    let p = PreFrames::from_v4_codes(codes, frames, bars);
    let no = AtomicBool::new(false);
    let py = Pyramid::build(&p, bars, &no).unwrap();
    let mut out = vec![0u16; bars];
    for d in [8.5f64, 16.5, 24.5] {
        let spans = spans(1700.0, 1700.0 + width as f64 * d, width, frames);
        let rows: usize = spans.iter().map(|&sp| py.rows_for(sp)).sum();
        let t = Instant::now();
        for _ in 0..10 { for &sp in &spans { py.column_max(&p, sp, &mut out); black_box(&out); } }
        let pyramid_ms = t.elapsed().as_secs_f64() * 100.0;
        let t = Instant::now();
        for _ in 0..10 { for &sp in &spans { column_max(&p, sp, &mut out); black_box(&out); } }
        let scan_ms = t.elapsed().as_secs_f64() * 100.0;
        let scan_rows: usize = spans.iter().map(|s| s.1-s.0).sum();
        println!("d={d} pyramid_ms={pyramid_ms:.4} scan_ms={scan_ms:.4} pyramid_rows={rows} scan_rows={scan_rows} normalized_ratio={:.4}",
            (pyramid_ms / rows as f64) / (scan_ms / scan_rows as f64));
    }
}

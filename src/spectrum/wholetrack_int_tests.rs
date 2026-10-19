//! The whole-track view through production entry points: real analyses on
//! private fixtures, the cache store, lazy channels, and the view drawn in a
//! headless egui context so its uploads, input and worker run as they do in
//! the window. Interleavings are forced with the worker's gate, never sleeps.
use super::*;
use crate::spectrum::palette::SpectrumPalette;
use crate::spectrum::wholetrack::{self as wt, Origin, Unavailable, View};
use crate::spectrum::wholetrack_ui::Plane;

const W: f32 = 900.0;
const H: f32 = 600.0;

fn input(t: f64, events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(W, H))),
        max_texture_side: Some(8192),
        time: Some(t),
        events,
        ..Default::default()
    }
}

struct Ui {
    ctx: egui::Context,
    t: f64,
    /// The window's width when not `W`.
    width: Option<f32>,
}

impl Ui {
    fn new() -> Self {
        Ui { ctx: egui::Context::default(), t: 0.0, width: None }
    }

    /// One frame of the window's whole-track host, with `events`.
    fn frame(&mut self, w: &mut SpectrumWindow, events: Vec<egui::Event>) {
        self.t += 1.0 / 60.0;
        let mut i = input(self.t, events);
        if let Some(w) = self.width {
            i.screen_rect = Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w, H)));
        }
        let _ = self.ctx.run(i, |c| {
            egui::CentralPanel::default().show(c, |ui| w.whole_track_ui(ui));
        });
    }

    /// One frame with these keys held (egui reads modifiers from the frame's
    /// input, as the platform layer reports them, not from the events).
    fn frame_held(&mut self, w: &mut SpectrumWindow, events: Vec<egui::Event>, modifiers: egui::Modifiers) {
        self.t += 1.0 / 60.0;
        let mut i = input(self.t, events);
        i.modifiers = modifiers;
        if let Some(w) = self.width {
            i.screen_rect = Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w, H)));
        }
        let _ = self.ctx.run(i, |c| {
            egui::CentralPanel::default().show(c, |ui| w.whole_track_ui(ui));
        });
    }

    /// Frames (paused) until the view shows a raster and nothing is pending.
    fn settle(&mut self, w: &mut SpectrumWindow) {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            frame(w, false);
            self.frame(w, vec![]);
            if !w.whole_track.shown().is_empty() && !w.whole_track.worker().busy() {
                // One more frame, so a result that just landed is the one drawn.
                self.frame(w, vec![]);
                if !w.whole_track.worker().busy() {
                    return;
                }
            }
            assert!(Instant::now() < deadline, "the whole-track view never settled");
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}

/// Direct reference for what a raster of `view` at `w × h` must hold.
fn reference(p: &cache::PreFrames, v: View, w: usize, h: usize, data_bars: usize) -> Vec<u32> {
    let mut out = vec![wt::OUTSIDE; w * h];
    for y in 0..h {
        let (g0, g1) = wt::span(v.b0, v.b1, h, h - 1 - y, p.bars());
        for x in 0..w {
            let (f0, f1) = wt::span(v.t0, v.t1, w, x, p.len());
            if f1 <= f0 || g1 <= g0 {
                continue;
            }
            if g0 >= data_bars {
                out[y * w + x] = wt::NO_DATA;
                continue;
            }
            let m = (f0..f1).flat_map(|f| p.row(f).unwrap()[g0..g1.min(data_bars)].to_vec()).max().unwrap();
            out[y * w + x] = m as u32;
        }
    }
    out
}

fn open_window(cache: &Path, stereo: bool) -> SpectrumWindow {
    let mut w = window(cache, 1.0);
    w.analyzer.pre_stereo_enabled = stereo;
    // The Spectrum window itself is open: a closed one shows nothing.
    w.open = true;
    w.whole_track.open = true;
    w
}

fn shown_matches(w: &mut SpectrumWindow, planes: &[&cache::PreFrames]) {
    let data_bars = w.whole_track.snapshot().unwrap().geometry.data_bars;
    let shown: Vec<_> = w.whole_track.shown();
    assert_eq!(shown.len(), planes.len(), "planes shown");
    for (r, p) in shown.iter().zip(planes) {
        assert_eq!(r.codes, reference(p, r.view, r.w, r.h, data_bars), "raster differs from a direct scan");
    }
}

/// Opening the view, fresh and from the store, shows the stored codes on the
/// record's geometry — and opening it starts no analysis of its own.
#[test]
fn fresh_and_reloaded_analyses_draw_the_same_cells_on_the_records_axes() {
    let d = Dir::new("wt_fresh");
    let track = tracks(&d, 1).remove(0);
    let cache = d.0.join("cache");
    // Fresh: the view is open before the analysis exists (paused arrival).
    let mut w1 = open_window(&cache, false);
    let mut u1 = Ui::new();
    u1.frame(&mut w1, vec![]);
    assert_eq!(w1.whole_track.snapshot().map(|_| ()), None);
    w1.on_play(&track, SR);
    assert!(settle(&mut w1));
    u1.settle(&mut w1);
    assert_eq!(spawned(&w1), 1, "the view must not analyse");
    let s1 = w1.whole_track.snapshot().unwrap();
    assert!(matches!(s1.origin, Origin::Store { .. }), "{:?}", s1.origin);
    assert!(s1.pin.is_some());
    let settings = w1.analyzer.effective_settings(&track);
    assert_eq!(s1.geometry, wt::Geometry::parse(&settings.record(SR)).unwrap());
    assert_eq!(s1.geometry.transform, wt::Transform::Superlet);
    assert_eq!(s1.geometry.axis_hi, 16_000.0);
    let g1 = s1.geometry;
    let fresh = (*w1.analyzer.pre_frames).clone();
    shown_matches(&mut w1, &[&fresh]);
    let r1 = w1.whole_track.shown()[0].clone();

    // Reloaded: an exact hit in a new window, nothing analysed.
    let mut w2 = open_window(&cache, false);
    let mut u2 = Ui::new();
    w2.on_play(&track, SR);
    assert!(settle(&mut w2));
    u2.settle(&mut w2);
    assert_eq!(spawned(&w2), 0, "a valid result was transformed again");
    assert_eq!(*w2.analyzer.pre_frames, fresh, "the reload holds other codes");
    assert_eq!(w2.whole_track.snapshot().unwrap().geometry, g1);
    let r2 = w2.whole_track.shown()[0].clone();
    assert_eq!((r2.w, r2.h, r2.view), (r1.w, r1.h, r1.view));
    assert_eq!(r2.codes, r1.codes, "fresh and reloaded pictures differ");
}

/// Left and Right come from their own analysis, lazily, under the same
/// generation; a Mix-only analysis offers them and never makes them up.
#[test]
fn channels_are_their_own_planes_and_are_never_derived_from_the_mix() {
    let d = Dir::new("wt_lr");
    let track = tracks(&d, 1).remove(0);
    // Mix only: Left is unavailable, and asking does not analyse.
    let mut w = open_window(&d.0.join("c1"), false);
    let mut u = Ui::new();
    w.on_play(&track, SR);
    assert!(settle(&mut w));
    w.whole_track.plane = Plane::Left;
    for _ in 0..20 {
        frame(&mut w, false);
        u.frame(&mut w, vec![]);
    }
    assert_eq!(spawned(&w), 1);
    assert!(w.whole_track.shown().is_empty(), "a Left picture appeared without Left data");
    assert!(w.whole_track.snapshot().unwrap().lr.is_none());

    // Stereo: Split shows the two stored planes, linked.
    let mut w = open_window(&d.0.join("c2"), true);
    let mut u = Ui::new();
    w.on_play(&track, SR);
    assert!(settle(&mut w));
    // A fresh stereo analysis installs its channels resident; a reload reads
    // them lazily. Use a reload, which is the lazy path.
    let mut w = open_window(&d.0.join("c2"), false);
    w.on_play(&track, SR);
    assert!(settle(&mut w));
    assert_eq!(spawned(&w), 0);
    assert!(!w.analyzer.pre_channels_resident(), "setup: channels should start on disk");
    w.whole_track.plane = Plane::Split;
    u.settle(&mut w);
    let pair = w.analyzer.pre_channels().resident().expect("the view did not acquire the channels").clone();
    assert_ne!(pair.left(), pair.right(), "setup: the fixture's channels should differ");
    shown_matches(&mut w, &[pair.left(), pair.right()]);
    let shown = w.whole_track.shown();
    assert_eq!(shown[0].view, shown[1].view, "Split is one time/frequency view");
    assert_ne!(shown[0].codes, shown[1].codes);
    // Back to Mix: the view lets the channels go.
    w.whole_track.plane = Plane::Mix;
    u.settle(&mut w);
    assert!(w.whole_track.snapshot().unwrap().lr.is_none());
    frame(&mut w, false);
    assert!(!w.analyzer.pre_channels_resident(), "the channels stayed resident for nobody");
}

/// A raster still running when the track changes is never shown, and the
/// new track's picture is.
#[test]
fn a_late_raster_for_the_previous_track_is_refused() {
    let d = Dir::new("wt_late");
    let t = tracks(&d, 2);
    let mut w = open_window(&d.0.join("c"), false);
    let mut u = Ui::new();
    w.on_play(&t[0], SR);
    assert!(settle(&mut w));
    u.settle(&mut w);
    let a = w.whole_track.shown()[0].clone();
    // Hold the next raster, and ask for one by zooming.
    let gate = Arc::new(AtomicBool::new(true));
    w.whole_track.sync_cells_for_test = Some(0);
    w.whole_track.worker().hold = Some(gate.clone());
    w.whole_track.plane = Plane::Mix;
    let v = w.whole_track.view().unwrap();
    w.whole_track.set_view_for_test(View { t1: v.t1 * 0.5, ..v });
    u.frame(&mut w, vec![]);
    assert!(w.whole_track.worker().busy(), "setup: no raster in flight");
    // Change track while it is held.
    w.on_play(&t[1], SR);
    assert!(settle(&mut w));
    gate.store(false, Ordering::SeqCst);
    w.whole_track.worker().hold = None;
    u.settle(&mut w);
    let b_frames = (*w.analyzer.pre_frames).clone();
    shown_matches(&mut w, &[&b_frames]);
    assert_ne!(w.whole_track.shown()[0].codes, a.codes);
    assert!(w.whole_track.worker().discarded >= 1 || w.whole_track.worker().started >= 2);
}

/// The view holds the generation pin and the plane only while it is open,
/// and closing it releases them without touching the analyser's own.
#[test]
fn closing_the_view_releases_its_pin_and_planes_only() {
    let d = Dir::new("wt_pin");
    let track = tracks(&d, 1).remove(0);
    let mut w = open_window(&d.0.join("c"), false);
    let mut u = Ui::new();
    w.on_play(&track, SR);
    assert!(settle(&mut w));
    let installed = match w.analyzer.pre_cache() {
        Some(CacheRef::Store { installed, .. }) => installed.clone(),
        other => panic!("setup: {other:?}"),
    };
    let base_pin = Arc::strong_count(&installed);
    u.settle(&mut w);
    assert_eq!(Arc::strong_count(&installed), base_pin + 1, "the snapshot does not hold the pin");
    assert!(Arc::strong_count(&w.analyzer.pre_frames) >= 2);
    w.whole_track.close();
    assert_eq!(Arc::strong_count(&installed), base_pin);
    let deadline = Instant::now() + Duration::from_secs(20);
    while Arc::strong_count(&w.analyzer.pre_frames) != 1 {
        assert!(Instant::now() < deadline, "a raster thread kept the plane alive");
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(!w.analyzer.pre_frames.is_empty(), "closing the view cleared the analysis");
    assert!(w.analyzer.pre_cache().is_some());
}

/// An analysis whose settings are not recorded is not drawn on guessed axes.
#[test]
fn an_analysis_without_a_record_is_not_drawn() {
    let d = Dir::new("wt_norec");
    let track = tracks(&d, 1).remove(0);
    let mut w = open_window(&d.0.join("c"), false);
    let mut u = Ui::new();
    // A recorded analysis first, so a record left behind would be visible.
    w.on_play(&track, SR);
    assert!(settle(&mut w));
    u.settle(&mut w);
    assert!(w.analyzer.pre_record().is_some());
    let started = w.whole_track.worker().started;
    let rows: Vec<Vec<f32>> = (0..50).map(|f| (0..BARS).map(|b| ((f + b) % 10) as f32 / 10.0).collect()).collect();
    w.analyzer.set_pre_frames(cache::PreFrames::from_analysis(&rows), 180.0);
    assert_eq!(w.analyzer.pre_record(), None);
    for _ in 0..5 {
        u.frame(&mut w, vec![]);
    }
    assert!(w.whole_track.snapshot().is_none());
    assert!(w.whole_track.shown().is_empty(), "the previous analysis is still drawn");
    assert_eq!(w.whole_track.unavailable_for_test(), Some(Unavailable::NoRecord));
    assert_eq!(w.whole_track.worker().started, started);
}

/// The playhead moving rasterises nothing; a palette change recolours
/// without rasterising; a click asks the player to seek, and a drag does not.
#[test]
fn playback_palette_and_clicks_cost_only_what_they_change() {
    let d = Dir::new("wt_cost");
    let track = tracks(&d, 1).remove(0);
    let mut w = open_window(&d.0.join("c"), false);
    let mut u = Ui::new();
    w.on_play(&track, SR);
    assert!(settle(&mut w));
    u.settle(&mut w);
    let (req, rec) = (w.whole_track.usage.requests, w.whole_track.usage.recolours);
    for i in 0..100 {
        w.tick(i as f64 * 0.01, true);
        u.frame(&mut w, vec![]);
    }
    assert_eq!(w.whole_track.usage.requests, req, "the playhead caused a raster");
    assert_eq!(w.whole_track.usage.recolours, rec, "the playhead caused a recolour");
    w.palette_kind = if w.palette_kind == SpectrumPalette::Ice { SpectrumPalette::Classic } else { SpectrumPalette::Ice };
    u.frame(&mut w, vec![]);
    assert_eq!(w.whole_track.usage.requests, req, "a palette change rasterised");
    assert_eq!(w.whole_track.usage.recolours, rec + 2, "main and overview should be recoloured once each");

    // All code colours follow palette and album-accent changes immediately;
    // reuse must never leave the old colour table behind.
    for kind in SpectrumPalette::ALL {
        for accent in [egui::Color32::RED, egui::Color32::GREEN] {
            w.palette_kind = kind;
            w.palette_accent = accent;
            u.frame(&mut w, vec![]);
            let snap = w.whole_track.snapshot().unwrap();
            let expected: Vec<_> = (0..=snap.max_code).map(|c| crate::spectrum::palette::Palette::new(kind, accent).heat(c as f32 * snap.inv_max)).collect();
            assert_eq!(w.whole_track.colours_for_test().unwrap(), expected);
            assert_eq!(w.whole_track.usage.requests, req);
        }
    }

    // Clicks inside the plot, left then right: two seeks, in order.
    let click = |x: f32, y: f32| {
        let p = egui::pos2(x, y);
        vec![
            egui::Event::PointerMoved(p),
            egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: true, modifiers: Default::default() },
            egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: false, modifiers: Default::default() },
        ]
    };
    assert_eq!(w.take_seek_request(), None);
    u.frame(&mut w, click(300.0, 300.0));
    let a = w.take_seek_request().expect("a click did not seek");
    u.t += 1.0;
    u.frame(&mut w, vec![]);
    u.frame(&mut w, click(700.0, 300.0));
    let b = w.take_seek_request().expect("a second click did not seek");
    let dur = w.analyzer.pre_frames.len() as f64 / w.analyzer.pre_frame_rate;
    assert!(0.0 <= a && a < b && b < dur, "{a} {b} {dur}");
    // A drag pans and does not seek.
    let p0 = egui::pos2(400.0, 300.0);
    let p1 = egui::pos2(500.0, 300.0);
    u.frame(&mut w, vec![
        egui::Event::PointerMoved(p0),
        egui::Event::PointerButton { pos: p0, button: egui::PointerButton::Primary, pressed: true, modifiers: Default::default() },
    ]);
    u.frame(&mut w, vec![egui::Event::PointerMoved(p1)]);
    u.frame(&mut w, vec![egui::Event::PointerButton { pos: p1, button: egui::PointerButton::Primary, pressed: false, modifiers: Default::default() }]);
    assert_eq!(w.take_seek_request(), None, "a drag seeked");
}

fn press(p: egui::Pos2, button: egui::PointerButton, pressed: bool) -> egui::Event {
    egui::Event::PointerButton { pos: p, button, pressed, modifiers: Default::default() }
}

/// The gesture contract, through real press/move/release events in the
/// headless context: a primary click seeks on its release and nothing else;
/// a double click is two seeks and never a Fit; a primary drag pans without
/// seeking; a middle click fits without seeking; a middle drag does neither;
/// clicks at the plot's edges seek inside the track and clicks beside it do
/// nothing.
#[test]
fn gestures_are_decided_on_their_own_release() {
    use egui::PointerButton::{Middle, Primary};
    let d = Dir::new("wt_gestures");
    let track = tracks(&d, 1).remove(0);
    let mut w = open_window(&d.0.join("c"), false);
    // The gestures alone; following has its own tests.
    w.whole_track.follow = false;
    let mut u = Ui::new();
    w.on_play(&track, SR);
    assert!(settle(&mut w));
    u.settle(&mut w);
    let frames = w.analyzer.pre_frames.len() as f64;
    let dur = frames / w.analyzer.pre_frame_rate;
    let full = w.whole_track.view().unwrap();
    // Zoomed in, so a Fit shows.
    let zoomed = View { t0: full.t1 * 0.25, t1: full.t1 * 0.5, ..full };
    w.whole_track.set_view_for_test(zoomed);
    u.frame(&mut w, vec![]);
    let plot = w.whole_track.plot_for_test.expect("setup: no plot laid out");
    let mid = plot.center();

    // Single click: one seek, on the release, and the view is untouched.
    u.frame(&mut w, vec![egui::Event::PointerMoved(mid), press(mid, Primary, true)]);
    assert_eq!(w.take_seek_request(), None, "a press alone seeked");
    u.frame(&mut w, vec![press(mid, Primary, false)]);
    let s = w.take_seek_request().expect("a click did not seek");
    let want = w.whole_track.snapshot().unwrap().geometry.frame_time(((zoomed.t0 + zoomed.t1) * 0.5).floor());
    assert!((s - want).abs() <= 1.0 / w.analyzer.pre_frame_rate + 1e-9, "seeked to {s}, centre is {want}");
    assert_eq!(w.whole_track.view(), Some(zoomed), "a click moved the view");

    // Double click (well inside the double-click interval): two seeks, no Fit.
    u.t += 1.0;
    u.frame(&mut w, vec![]);
    let mut seeks = 0;
    for _ in 0..2 {
        u.frame(&mut w, vec![press(mid, Primary, true)]);
        u.frame(&mut w, vec![press(mid, Primary, false)]);
        seeks += w.take_seek_request().is_some() as usize;
    }
    assert_eq!(seeks, 2, "each click of a double click seeks on its own release");
    assert_eq!(w.whole_track.view(), Some(zoomed), "a double click fitted or moved the view");

    // Primary drag: pans, does not seek.
    u.t += 1.0;
    u.frame(&mut w, vec![]);
    let to = mid + egui::vec2(-120.0, 0.0);
    u.frame(&mut w, vec![press(mid, Primary, true)]);
    u.frame(&mut w, vec![egui::Event::PointerMoved(mid + egui::vec2(-60.0, 0.0))]);
    u.frame(&mut w, vec![egui::Event::PointerMoved(to)]);
    u.frame(&mut w, vec![press(to, Primary, false)]);
    assert_eq!(w.take_seek_request(), None, "a drag seeked");
    let panned = w.whole_track.view().unwrap();
    assert!(panned.t0 > zoomed.t0, "a leftward drag did not pan later in time: {panned:?}");
    assert!((panned.t1 - panned.t0 - (zoomed.t1 - zoomed.t0)).abs() < 1e-9, "a drag changed the zoom");

    // Middle drag: neither pans, nor fits, nor seeks.
    u.t += 1.0;
    u.frame(&mut w, vec![egui::Event::PointerMoved(mid)]);
    u.frame(&mut w, vec![press(mid, Middle, true)]);
    u.frame(&mut w, vec![egui::Event::PointerMoved(to)]);
    u.frame(&mut w, vec![press(to, Middle, false)]);
    assert_eq!(w.take_seek_request(), None, "a middle drag seeked");
    assert_eq!(w.whole_track.view(), Some(panned), "a middle drag changed the view");

    // Middle click: fits the whole track, does not seek.
    u.t += 1.0;
    u.frame(&mut w, vec![egui::Event::PointerMoved(mid)]);
    u.frame(&mut w, vec![press(mid, Middle, true)]);
    u.frame(&mut w, vec![press(mid, Middle, false)]);
    assert_eq!(w.take_seek_request(), None, "a middle click seeked");
    assert_eq!(w.whole_track.view(), Some(full), "a middle click did not fit the whole track");

    // Edges: the plot's first and last points seek inside the track; a click
    // just beside the plot (the axis) does nothing.
    for (x, inside) in [(plot.left() + 0.5, true), (plot.right() - 0.5, true), (plot.left() - 3.0, false)] {
        u.t += 1.0;
        let p = egui::pos2(x, mid.y);
        u.frame(&mut w, vec![egui::Event::PointerMoved(p)]);
        u.frame(&mut w, vec![press(p, Primary, true)]);
        u.frame(&mut w, vec![press(p, Primary, false)]);
        match (w.take_seek_request(), inside) {
            (Some(s), true) => assert!((0.0..dur).contains(&s), "edge click at {x} seeked to {s} of {dur}"),
            (None, false) => {}
            (got, _) => panic!("click at x={x}: {got:?}"),
        }
    }
    assert_eq!(w.whole_track.view(), Some(full), "edge clicks moved the view");
}

// ── Repeats ───────────────────────────────────────────────────────────────

fn installed_of(w: &SpectrumWindow) -> Arc<store::Installed> {
    match w.analyzer.pre_cache() {
        Some(CacheRef::Store { installed, .. }) => installed.clone(),
        other => panic!("setup: not a store install: {other:?}"),
    }
}

/// A shown, settled view of `track` at a zoom that is not the whole track.
fn shown_zoomed(cache: &Path, track: &Path) -> (SpectrumWindow, Ui, View) {
    let mut w = open_window(cache, false);
    w.whole_track.follow = false;
    let mut u = Ui::new();
    w.on_play(track, SR);
    assert!(settle(&mut w));
    u.settle(&mut w);
    let full = w.whole_track.view().unwrap();
    let zoomed = View { t0: full.t1 * 0.2, t1: full.t1 * 0.45, b0: 3.0, b1: full.b1 - 5.0 };
    w.whole_track.set_view_for_test(zoomed);
    u.settle(&mut w);
    (w, u, zoomed)
}

/// The same track started again (a repeat, or a restart from the top) after
/// its analysis finished: through the production entry point, playback
/// restarts and the analysis does not. The same planes and store pin stay
/// installed (nothing is read or decoded), the revision does not move, the
/// whole-track view keeps its zoom and frequency span and asks for no new
/// raster, and only the transport state (playhead, cursor) starts over.
#[test]
fn a_repeat_keeps_the_finished_analysis_and_the_view() {
    let d = Dir::new("wt_repeat");
    let track = tracks(&d, 1).remove(0);
    let (mut w, mut u, zoomed) = shown_zoomed(&d.0.join("c"), &track);
    let frames = w.analyzer.pre_frames.clone();
    let installed = installed_of(&w);
    let rev = w.analyzer.pre_revision();
    let (req, rec) = (w.whole_track.usage.requests, w.whole_track.usage.recolours);
    let shown = w.whole_track.shown()[0].clone();
    for i in 0..30 {
        w.tick(0.3 + i as f64 * 0.01, true);
        u.frame(&mut w, vec![]);
    }
    let spawned_before = spawned(&w);
    w.on_play_live(&track, SR, Some(SR));
    w.tick(0.0, true);
    u.frame(&mut w, vec![]);
    assert!(Arc::ptr_eq(&frames, &w.analyzer.pre_frames), "the repeat installed other planes (a reload)");
    assert!(Arc::ptr_eq(&installed, &installed_of(&w)), "the repeat took another pin");
    assert_eq!(w.analyzer.pre_revision(), rev, "the repeat moved the revision");
    assert_eq!(spawned(&w), spawned_before, "the repeat analysed");
    assert!(w.analyzer.analysis_idle(), "the repeat queued work");
    assert_eq!(w.whole_track.view(), Some(zoomed), "the repeat reset the zoom");
    assert_eq!(w.whole_track.usage.requests, req, "the repeat asked for a raster");
    assert_eq!(w.whole_track.usage.recolours, rec, "the repeat recoloured");
    assert_eq!(w.whole_track.shown()[0], shown);
    assert_eq!(w.whole_track.playhead, 0.0);
    assert!(w.waveform.is_some() || w.analyzer.pending_waveform.is_some(), "the repeat dropped the waveform");

    // Following: the repeat scrolls back to the start at the same zoom and
    // frequency span, and keeps the planes all the same.
    w.whole_track.follow = true;
    for i in 0..30 {
        w.tick(0.5 + i as f64 * 0.01, true);
        u.frame(&mut w, vec![]);
    }
    let before = w.whole_track.view().unwrap();
    w.on_play_live(&track, SR, Some(SR));
    w.tick(0.0, true);
    u.frame(&mut w, vec![]);
    let after = w.whole_track.view().unwrap();
    assert!(Arc::ptr_eq(&frames, &w.analyzer.pre_frames), "a following repeat reloaded");
    assert_eq!(after.t0, 0.0, "following did not go back to the start");
    assert!((after.t1 - after.t0 - (before.t1 - before.t0)).abs() < 1e-9, "the time span changed");
    assert_eq!((after.b0, after.b1), (before.b0, before.b1), "the frequency span changed");
}

/// Following: every frame the time window moves with the playhead, keeping
/// it a third of the way across and the zoom and frequency window as they
/// were; each step is the playhead's own movement (no page jumps); every
/// frame's picture is exactly the camera drawn. Pause holds it; a drag sets
/// following aside until it resumes; a time zoom (wheel) and a frequency zoom
/// (Shift+wheel) do not — the time zoom is around the playhead while
/// following and around the pointer otherwise; a click seeks and following
/// carries on from there.
#[test]
fn following_scrolls_with_the_playhead_exactly_and_yields_to_the_hand() {
    let d = Dir::new("wt_follow");
    let track = tracks(&d, 1).remove(0);
    let mut w = open_window(&d.0.join("c"), false);
    let mut u = Ui::new();
    w.on_play(&track, SR);
    assert!(settle(&mut w));
    u.settle(&mut w);
    assert!(w.whole_track.follow, "following is not the default");
    let full = w.whole_track.view().unwrap();
    let rate = w.analyzer.pre_frame_rate;
    let zoomed = View { t0: 0.0, t1: full.t1 * 0.25, b0: 2.0, b1: full.b1 - 3.0 };
    w.whole_track.set_view_for_test(zoomed);
    let span_t = zoomed.t1 - zoomed.t0;
    let mut last: Option<View> = None;
    for i in 0..40 {
        let ph = 0.2 + i as f64 / 60.0;
        w.tick(ph, true);
        u.frame(&mut w, vec![]);
        let cam = w.whole_track.exact_camera().expect("a following frame was not exact");
        let shown = w.whole_track.shown();
        let data_bars = w.whole_track.snapshot().unwrap().geometry.data_bars;
        assert_eq!(shown[0].codes, reference(&w.analyzer.pre_frames, cam, shown[0].w, shown[0].h, data_bars), "frame {i}");
        let px = span_t / shown[0].w as f64;
        let want = (ph * rate - span_t / 3.0).clamp(0.0, full.t1 - span_t);
        assert!((cam.t0 - want).abs() <= px, "frame {i}: camera at {} wants {want}", cam.t0);
        // The frequency window keeps its span and sits on the row grid,
        // within half a row of where it was asked to be.
        let row = (zoomed.b1 - zoomed.b0) / shown[0].h as f64;
        assert!((cam.t1 - cam.t0 - span_t).abs() < 1e-6 && (cam.b1 - cam.b0 - (zoomed.b1 - zoomed.b0)).abs() < 1e-9
            && (cam.b0 - zoomed.b0).abs() <= row / 2.0 + 1e-9, "frame {i}: zoom changed");
        if let Some(l) = last {
            let step = (cam.t0 - l.t0) / px;
            let moved = rate / 60.0 / px;
            assert!(step >= 0.0 && (step - moved).abs() <= 1.0 + 1e-6 || cam.t0 == full.t1 - span_t, "frame {i}: stepped {step} px for {moved}");
        }
        last = Some(cam);
    }
    // Paused: the camera holds.
    let held = w.whole_track.view();
    for _ in 0..5 {
        w.tick(0.2 + 39.0 / 60.0, false);
        u.frame(&mut w, vec![]);
    }
    assert_eq!(w.whole_track.view(), held, "a paused playhead moved the camera");
    // Shift+wheel is not inspection of time: following carries on.
    let mid = w.whole_track.plot_for_test.unwrap().center();
    let shift = egui::Modifiers { shift: true, ..Default::default() };
    let before = w.whole_track.view().unwrap();
    u.frame_held(&mut w, vec![egui::Event::PointerMoved(mid), egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: egui::vec2(0.0, 40.0), modifiers: shift }], shift);
    let after = w.whole_track.view().unwrap();
    assert!((after.b1 - after.b0) < (before.b1 - before.b0), "setup: Shift+wheel did not zoom frequency");
    assert!((after.t1 - after.t0 - span_t).abs() < 1e-6, "Shift+wheel zoomed time");
    assert!(!w.whole_track.follow_suspended, "Shift+wheel set following aside");
    // A drag does set it aside, and the camera then stays where it was put.
    w.whole_track.set_view_for_test(View { t0: 10.0, t1: 10.0 + span_t, ..w.whole_track.view().unwrap() });
    u.frame(&mut w, vec![press(mid, egui::PointerButton::Primary, true)]);
    u.frame(&mut w, vec![egui::Event::PointerMoved(mid + egui::vec2(-80.0, 0.0))]);
    u.frame(&mut w, vec![press(mid + egui::vec2(-80.0, 0.0), egui::PointerButton::Primary, false)]);
    assert!(w.whole_track.follow_suspended, "a drag did not set following aside");
    let put = w.whole_track.view();
    for i in 0..5 {
        w.tick(0.5 + i as f64 / 60.0, true);
        u.frame(&mut w, vec![]);
    }
    assert_eq!(w.whole_track.view(), put, "a suspended follow moved the camera");
    assert!(w.whole_track.follow, "suspending changed the preference");
    // Resume follow (the button's action): back on the playhead, same zoom.
    w.whole_track.follow_suspended = false;
    w.tick(0.55, true);
    u.frame(&mut w, vec![]);
    let cam = w.whole_track.exact_camera().unwrap();
    assert!((cam.t0 - (0.55 * rate - span_t / 3.0)).abs() <= span_t, "resuming did not return to the playhead");
    assert!((cam.t1 - cam.t0 - span_t).abs() < 1e-6, "resuming changed the zoom");
    // A time zoom with the wheel while following: around the playhead, which
    // keeps its place a third of the way across, and following carries on
    // (the owner's 1.5.4 policy; it used to set following aside). Nothing
    // seeks, and the frequency window stays.
    let wheel = |dy: f32| vec![egui::Event::PointerMoved(mid + egui::vec2(150.0, 0.0)),
        egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: egui::vec2(0.0, dy), modifiers: Default::default() }];
    let before = w.whole_track.view().unwrap();
    u.frame(&mut w, wheel(40.0));
    let after = w.whole_track.view().unwrap();
    assert!(after.t1 - after.t0 < (before.t1 - before.t0) * 0.99, "setup: the wheel did not zoom time");
    assert!(!w.whole_track.follow_suspended, "a wheel zoom set following aside");
    assert!(w.take_seek_request().is_none(), "a wheel zoom sought");
    assert_eq!((after.b0, after.b1), (before.b0, before.b1), "a time zoom changed the frequency window");
    let at = |v: View, ph: f64| (ph * rate - v.t0) / (v.t1 - v.t0);
    let ph_now = 0.55;
    assert!((at(after, ph_now) - 1.0 / 3.0).abs() < 0.01, "the playhead moved to {:.3} of the window", at(after, ph_now));
    let zspan = after.t1 - after.t0;
    for i in 1..=10 {
        let ph = ph_now + i as f64 / 60.0;
        w.tick(ph, true);
        u.frame(&mut w, vec![]);
        let v = w.whole_track.view().unwrap();
        assert!((v.t1 - v.t0 - zspan).abs() < 1e-6, "frame {i}: the zoom did not hold");
        assert!((at(v, ph) - 1.0 / 3.0).abs() < 0.01, "frame {i}: not following after the zoom");
    }
    // Zooming back out keeps following too, and so does the Time − button's
    // policy (exercised through the same anchor, `following`).
    u.frame(&mut w, wheel(-40.0));
    assert!(!w.whole_track.follow_suspended, "zooming out set following aside");
    let v = w.whole_track.view().unwrap();
    assert!((at(v, ph_now + 10.0 / 60.0) - 1.0 / 3.0).abs() < 0.01, "zooming out moved the playhead");
    // Set aside (a drag), the wheel zooms around the pointer instead: the
    // time under the pointer stays where it is.
    u.frame(&mut w, vec![press(mid, egui::PointerButton::Primary, true)]);
    u.frame(&mut w, vec![egui::Event::PointerMoved(mid + egui::vec2(-40.0, 0.0))]);
    u.frame(&mut w, vec![press(mid + egui::vec2(-40.0, 0.0), egui::PointerButton::Primary, false)]);
    assert!(w.whole_track.follow_suspended, "setup: the drag did not set following aside");
    u.frame(&mut w, vec![]);
    let plot = w.whole_track.plot_for_test.unwrap();
    let px = mid.x + 150.0;
    let t_under = |v: View| v.t0 + ((px - plot.left()) / plot.width()) as f64 * (v.t1 - v.t0);
    let before = w.whole_track.view().unwrap();
    u.frame(&mut w, wheel(40.0));
    let after = w.whole_track.view().unwrap();
    assert!(after.t1 - after.t0 < before.t1 - before.t0, "setup: no zoom while set aside");
    let one_px = (before.t1 - before.t0) / plot.width() as f64;
    assert!((t_under(after) - t_under(before)).abs() <= 2.0 * one_px, "a set-aside zoom was not around the pointer");
    assert!(w.whole_track.follow_suspended, "the zoom resumed following by itself");
    // A click seeks without setting it aside.
    w.whole_track.follow_suspended = false;
    u.t += 1.0;
    u.frame(&mut w, vec![press(mid, egui::PointerButton::Primary, true)]);
    u.frame(&mut w, vec![press(mid, egui::PointerButton::Primary, false)]);
    assert!(w.take_seek_request().is_some(), "a click did not seek while following");
    assert!(!w.whole_track.follow_suspended, "a click set following aside");
}

/// What a repeat must NOT keep: each case is the same entry point after a
/// finished, shown analysis, and each must end with other planes installed
/// (or none), never the retained ones.
#[test]
fn a_repeat_does_not_keep_an_analysis_that_no_longer_applies() {
    // Another track with the same geometry: replaced, and fitted afresh.
    {
        let d = Dir::new("wt_rep_other");
        let t = tracks(&d, 2);
        let (mut w, mut u, _) = shown_zoomed(&d.0.join("c"), &t[0]);
        let old = w.analyzer.pre_frames.clone();
        w.on_play_live(&t[1], SR, Some(SR));
        assert!(settle(&mut w));
        u.settle(&mut w);
        assert!(!Arc::ptr_eq(&old, &w.analyzer.pre_frames));
        assert_eq!(old.bars(), w.analyzer.pre_frames.bars(), "setup: the geometry should match");
        assert_eq!(w.whole_track.view(), Some(View::full(w.analyzer.pre_frames.len(), BARS)));
    }
    // The same path with other content: the descriptor no longer matches, so
    // the file is analysed again.
    {
        let d = Dir::new("wt_rep_content");
        let t = tracks(&d, 1);
        let (mut w, mut u, _) = shown_zoomed(&d.0.join("c"), &t[0]);
        let old = w.analyzer.pre_frames.clone();
        let n = spawned(&w);
        std::thread::sleep(Duration::from_millis(20));
        write_wav(&t[0], &[880.0, 5000.0], 99);
        w.on_play_live(&t[0], SR, Some(SR));
        assert!(settle(&mut w));
        u.settle(&mut w);
        assert!(!Arc::ptr_eq(&old, &w.analyzer.pre_frames), "changed content kept the old planes");
        assert_eq!(spawned(&w), n + 1, "changed content was not analysed again");
        assert_ne!(*old, *w.analyzer.pre_frames);
    }
    // Other settings: the installed record no longer applies.
    {
        let d = Dir::new("wt_rep_settings");
        let t = tracks(&d, 1);
        let (mut w, mut u, _) = shown_zoomed(&d.0.join("c"), &t[0]);
        let old = w.analyzer.pre_frames.clone();
        w.analyzer.aslt_cfg = cfg(1.5);
        w.on_play_live(&t[0], SR, Some(SR));
        assert!(settle(&mut w));
        u.settle(&mut w);
        assert!(!Arc::ptr_eq(&old, &w.analyzer.pre_frames), "other settings kept the old planes");
    }
    // A newer generation published for the same stem (another window's
    // Re-analyze): the repeat installs the head, not the generation it had.
    {
        let d = Dir::new("wt_rep_gen");
        let t = tracks(&d, 1);
        let cache = d.0.join("c");
        let (mut w, mut u, _) = shown_zoomed(&cache, &t[0]);
        let old_gen = installed_of(&w).generation;
        let mut other = open_window(&cache, false);
        other.on_play(&t[0], SR);
        assert!(settle(&mut other));
        assert!(other.reanalyze());
        assert!(settle(&mut other));
        let new_gen = installed_of(&other).generation;
        assert_ne!(new_gen, old_gen, "setup: no newer generation");
        w.on_play_live(&t[0], SR, Some(SR));
        assert!(settle(&mut w));
        u.settle(&mut w);
        assert_eq!(installed_of(&w).generation, new_gen, "the repeat kept a superseded generation");
    }
    // A chosen alternate (a legacy file adopted for this track, as it can
    // be while the exact analysis is not shown): the repeat re-adopts it
    // through its own path, and the exact-hit retention never keeps it.
    {
        let d = Dir::new("wt_rep_alt");
        let t = tracks(&d, 1);
        let mut w = open_window(&d.0.join("c"), false);
        let (_, legacy) = w.legacy_key_with(&t[0], &w.analyzer.aslt_cfg.clone());
        let rows = cache::PreFrames::from_analysis(&vec![vec![0.5f32; BARS]; 40]);
        cache::write_atomic(&legacy, &rows.to_v4()).unwrap();
        let hold = park(&mut w);
        w.on_play(&t[0], SR);
        w.accept_alternate(&legacy).expect("setup: legacy refused");
        release(&mut w, &hold);
        assert!(settle(&mut w));
        assert_eq!(w.analyzer.pre_cache_path(), Some(legacy.as_path()), "setup: the alternate is not shown");
        let adopted = w.analyzer.pre_frames.clone();
        w.on_play_live(&t[0], SR, Some(SR));
        assert!(settle(&mut w));
        assert_eq!(w.analyzer.pre_cache_path(), Some(legacy.as_path()), "the repeat dropped the chosen alternate");
        assert!(!Arc::ptr_eq(&adopted, &w.analyzer.pre_frames), "an alternate was kept by the exact-hit retention");
    }
}

// ── Closing the real parent window ─────────────────────────────────────────

/// A stereo track, analysed and cached, shown in a fresh window reading its
/// channels lazily, with the view on `plane` and settled.
fn shown_stereo(d: &Dir, track: &Path, plane: Plane) -> (SpectrumWindow, Ui) {
    let cache = d.0.join("c");
    let mut w = open_window(&cache, true);
    w.on_play(track, SR);
    assert!(settle(&mut w));
    let mut w = open_window(&cache, false);
    w.open = true;
    let mut u = Ui::new();
    w.on_play(track, SR);
    assert!(settle(&mut w));
    w.whole_track.plane = plane;
    u.settle(&mut w);
    (w, u)
}

/// One frame of the whole window, as the player draws it (`tick`, then
/// `show`), with `close` as the native close box.
fn window_frame(w: &mut SpectrumWindow, u: &mut Ui, close: bool) {
    frame(w, false);
    u.t += 1.0 / 60.0;
    let mut input = input(u.t, vec![]);
    if close {
        let mut info = egui::ViewportInfo::default();
        info.events.push(egui::ViewportEvent::Close);
        input.viewports.insert(egui::ViewportId::ROOT, info);
    }
    let _ = u.ctx.run(input, |c| w.show(c));
}

fn wait_planes_released(p: &Arc<cache::PreFrames>, others: usize) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while Arc::strong_count(p) > others {
        assert!(Instant::now() < deadline, "a hidden view or its worker kept the plane: {} refs", Arc::strong_count(p));
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// The native close box and the main toolbar both close the window by
/// clearing `open`. Either way the hidden view keeps no planes, pin, rasters
/// or channel demand; the preference survives, and reopening shows the
/// track installed then.
#[test]
fn closing_the_spectrum_window_releases_the_view_by_either_route() {
    let d = Dir::new("wt_close");
    let track = tracks(&d, 1).remove(0);
    for route in ["native close box", "main toolbar"] {
        let (mut w, mut u) = shown_stereo(&d, &track, Plane::Split);
        let pair = w.analyzer.pre_channels().resident().expect("setup: channels resident").shared();
        let installed = match w.analyzer.pre_cache() {
            Some(CacheRef::Store { installed, .. }) => installed.clone(),
            other => panic!("setup: {other:?}"),
        };
        let pins = Arc::strong_count(&installed);
        match route {
            "native close box" => window_frame(&mut w, &mut u, true),
            _ => w.open = false,
        }
        assert!(!w.open, "{route}: the window did not close");
        // The player keeps ticking a closed window.
        for _ in 0..5 {
            window_frame(&mut w, &mut u, false);
        }
        assert!(w.whole_track.open, "{route}: closing the window turned the preference off");
        assert!(!w.whole_track.holds_anything(), "{route}: the hidden view still holds data");
        assert_eq!(Arc::strong_count(&installed), pins - 1, "{route}: the hidden view still pins the generation");
        assert!(!w.analyzer.pre_channels_wanted, "{route}: a hidden view still asks for channels");
        assert!(!w.analyzer.pre_channels_resident(), "{route}: the channels stayed resident for a hidden view");
        drop(installed);
        wait_planes_released(&pair.0, 1);
        wait_planes_released(&pair.1, 1);
        assert!(!w.analyzer.pre_frames.is_empty(), "{route}: closing cleared the analysis");
        // Reopen: the preference brings the view back on what is installed now.
        w.open = true;
        u.settle(&mut w);
        assert_eq!(w.whole_track.plane, Plane::Split);
        assert_eq!(w.whole_track.shown().len(), 2, "{route}: reopening did not show the view again");
    }
}

/// Closing while a raster is in flight: the result is never installed, and
/// the thread lets the plane go once it exits.
#[test]
fn closing_with_a_raster_in_flight_never_installs_it() {
    let d = Dir::new("wt_close_flight");
    let track = tracks(&d, 1).remove(0);
    let mut w = open_window(&d.0.join("c"), false);
    let mut u = Ui::new();
    w.on_play(&track, SR);
    assert!(settle(&mut w));
    u.settle(&mut w);
    let gate = Arc::new(AtomicBool::new(true));
    w.whole_track.sync_cells_for_test = Some(0);
    w.whole_track.worker().hold = Some(gate.clone());
    let v = w.whole_track.view().unwrap();
    w.whole_track.set_view_for_test(View { t1: v.t1 * 0.5, ..v });
    u.frame(&mut w, vec![]);
    assert!(w.whole_track.worker().busy(), "setup: nothing in flight");
    w.open = false;
    frame(&mut w, false);
    gate.store(false, Ordering::SeqCst);
    for _ in 0..200 {
        frame(&mut w, false);
        if !w.whole_track.holds_anything() {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(!w.whole_track.holds_anything());
    assert!(w.whole_track.shown().is_empty(), "a raster landed in a closed view");
    assert!(w.whole_track.worker().take_queued().is_none(), "a completion is still queued in a closed view");
    wait_planes_released(&w.analyzer.pre_frames.clone(), 2);
}

/// Changing track while the window is closed keeps nothing of the old one,
/// and reopening shows only the new one.
#[test]
fn a_track_changed_while_closed_is_the_only_one_shown_on_reopen() {
    let d = Dir::new("wt_close_track");
    let t = tracks(&d, 2);
    let mut w = open_window(&d.0.join("c"), false);
    let mut u = Ui::new();
    w.on_play(&t[0], SR);
    assert!(settle(&mut w));
    u.settle(&mut w);
    let old = w.analyzer.pre_frames.clone();
    w.open = false;
    frame(&mut w, false);
    w.on_play(&t[1], SR);
    assert!(settle(&mut w));
    // Only the test's handle holds the old song now.
    wait_planes_released(&old, 1);
    w.open = true;
    u.settle(&mut w);
    let b = (*w.analyzer.pre_frames).clone();
    shown_matches(&mut w, &[&b]);
    assert_ne!(*old, b, "setup: the two fixtures should differ");
}

/// Leaving Pre-process releases the view but keeps it selected; coming
/// back shows it again. The main spectrum's own channel view is another
/// consumer: its channels stay resident when the whole-track view goes.
#[test]
fn leaving_preprocess_releases_the_view_but_not_another_consumers_channels() {
    let d = Dir::new("wt_mode");
    let track = tracks(&d, 1).remove(0);
    let (mut w, mut u) = shown_stereo(&d, &track, Plane::Left);
    w.mode = SpectrumMode::RealTime;
    frame(&mut w, false);
    assert!(w.whole_track.open && !w.whole_track.holds_anything());
    w.mode = SpectrumMode::PreProcess;
    u.settle(&mut w);
    assert!(!w.whole_track.shown().is_empty());
    // Another consumer: the main plot's Split channel view.
    w.channel_view = channels::ChannelView::Split;
    w.style = VizStyle::Bars;
    frame(&mut w, false);
    assert!(w.analyzer.pre_channels_wanted);
    w.open = false;
    for _ in 0..3 {
        frame(&mut w, false);
    }
    assert!(!w.whole_track.holds_anything());
    assert!(w.analyzer.pre_channels_resident(), "the whole-track view's release took another consumer's channels");
}

/// Re-analyze while the view is open: the old snapshot goes with the old
/// analysis, and the new one is shown when it lands.
#[test]
fn a_replaced_analysis_replaces_the_view() {
    let d = Dir::new("wt_reanalyze");
    let track = tracks(&d, 1).remove(0);
    let mut w = open_window(&d.0.join("c"), false);
    let mut u = Ui::new();
    w.on_play(&track, SR);
    assert!(settle(&mut w));
    u.settle(&mut w);
    let old = w.analyzer.pre_frames.clone();
    assert!(w.reanalyze());
    u.frame(&mut w, vec![]);
    assert!(w.whole_track.snapshot().is_none(), "the cleared analysis is still the view's");
    assert!(settle(&mut w));
    u.settle(&mut w);
    assert!(!Arc::ptr_eq(&old, &w.whole_track.snapshot().unwrap().mix));
    let new = (*w.analyzer.pre_frames).clone();
    shown_matches(&mut w, &[&new]);
}

// ── The async path: what is drawn while the worker reads ─────────────────

/// What was drawn last frame, checked against the frozen reference for the
/// camera the frame asked for (not for any camera a ring held before): every
/// column drawn as data is that camera's reference column. Returns how many
/// columns were drawn as loading.
fn drawn_honest(w: &SpectrumWindow, planes: &[&cache::PreFrames], what: &str) -> usize {
    let db = w.whole_track.snapshot().unwrap().geometry.data_bars;
    let drawn = w.whole_track.drawn();
    assert_eq!(drawn.len(), planes.len(), "{what}: planes drawn");
    let mut loading = 0;
    for ((v, rw, rh, cols), p) in drawn.into_iter().zip(planes) {
        let want = reference(p, v, rw, rh, db);
        for (x, c) in cols.iter().enumerate() {
            match c {
                None => loading += 1,
                Some(c) => {
                    for y in 0..rh {
                        assert_eq!(c[y], want[y * rw + x], "{what}: column {x}, row {y} is drawn as data but is not the painted camera's");
                    }
                }
            }
        }
    }
    loading
}

/// The camera the view asked for this frame, on the pixel grid.
fn asked(w: &SpectrumWindow) -> View {
    w.whole_track.requested_for_test.expect("nothing requested")
}

fn hold_worker(w: &mut SpectrumWindow) -> Arc<AtomicBool> {
    let gate = Arc::new(AtomicBool::new(true));
    w.whole_track.worker().hold = Some(gate.clone());
    gate
}

fn release_worker(w: &mut SpectrumWindow, gate: &Arc<AtomicBool>) {
    gate.store(false, Ordering::SeqCst);
    w.whole_track.worker().hold = None;
}

/// Frames until every column of the asked camera is drawn as data; how many
/// frames that took. Each frame on the way is checked as honest.
fn until_exact(w: &mut SpectrumWindow, u: &mut Ui, planes: &[&cache::PreFrames], what: &str) -> usize {
    let deadline = Instant::now() + Duration::from_secs(60);
    for n in 1.. {
        u.frame(w, vec![]);
        drawn_honest(w, planes, what);
        if w.whole_track.exact_camera() == Some(asked(w)) {
            return n;
        }
        assert!(Instant::now() < deadline, "{what}: never became exact");
        std::thread::sleep(Duration::from_millis(2));
    }
    unreachable!()
}

/// Every camera change sent to the worker (nothing read on the UI thread),
/// with the worker held: a time zoom while following, a large seek while
/// following, a fast drag, and a completion superseded by a newer camera.
/// While the worker reads, the painted camera is the one asked for, its
/// columns are its own reference or drawn as loading — never an older
/// camera's picture moved or scaled into place — and once the worker lands
/// the camera is exact.
#[test]
fn async_camera_changes_retain_a_complete_honest_presentation() {
    let d = Dir::new("wt_async");
    let track = tracks(&d, 1).remove(0);
    let mut w = open_window(&d.0.join("c"), false);
    let mut u = Ui::new();
    w.on_play(&track, SR);
    assert!(settle(&mut w));
    u.settle(&mut w);
    let plane = (*w.analyzer.pre_frames).clone();
    let planes = [&plane];
    let rate = w.analyzer.pre_frame_rate;
    let full = w.whole_track.view().unwrap();
    w.whole_track.set_view_for_test(View { t0: 0.0, t1: full.t1 * 0.25, ..full });
    w.tick(0.3, true);
    u.settle(&mut w);
    assert!(w.whole_track.follow && !w.whole_track.follow_suspended, "setup: not following");
    w.whole_track.sync_cells_for_test = Some(0);
    let mid = w.whole_track.plot_for_test.unwrap().center();
    let previous = w.whole_track.painted_for_test.unwrap();

    // A time zoom while following: everything is new, so all of it loads.
    let gate = hold_worker(&mut w);
    w.tick(0.32, true);
    u.frame(&mut w, vec![egui::Event::PointerMoved(mid),
        egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: egui::vec2(0.0, 40.0), modifiers: Default::default() }]);
    assert!(!w.whole_track.follow_suspended, "the zoom set following aside");
    let loading = drawn_honest(&w, &planes, "time zoom, worker held");
    assert_eq!(loading, 0, "a ready picture was replaced by loading");
    assert_eq!(w.whole_track.painted_for_test, Some(previous), "the previous image was relabelled as the new camera");
    assert_ne!(asked(&w), previous, "setup: zoom did not change demand");
    assert!(w.whole_track.exact_camera().is_none(), "a retained camera counted as completed demand");
    // The playhead keeps moving while the worker is held: still honest.
    for i in 0..5 {
        w.tick(0.33 + i as f64 / 60.0, true);
        u.frame(&mut w, vec![]);
        drawn_honest(&w, &planes, "following while held");
    }
    release_worker(&mut w, &gate);
    let n = until_exact(&mut w, &mut u, &planes, "time zoom");
    assert_eq!(drawn_honest(&w, &planes, "time zoom, landed"), 0);
    eprintln!("time zoom: exact after {n} frame(s) once released");

    // A large seek while following: the new place is not held.
    let gate = hold_worker(&mut w);
    let far = 0.8 * full.t1 / rate;
    w.tick(far, true);
    u.frame(&mut w, vec![]);
    let v = asked(&w);
    assert!((far * rate - v.t0) / (v.t1 - v.t0) - 1.0 / 3.0 < 0.01, "setup: the camera did not follow the seek");
    assert_eq!(drawn_honest(&w, &planes, "seek, worker held"), 0, "seek discarded the complete presentation");
    assert!(w.whole_track.exact_camera().is_none());
    release_worker(&mut w, &gate);
    until_exact(&mut w, &mut u, &planes, "seek");

    // A fast drag (sets following aside): what overlaps the old camera with
    // the same spans is data, the rest loads.
    let gate = hold_worker(&mut w);
    u.frame(&mut w, vec![press(mid, egui::PointerButton::Primary, true)]);
    u.frame(&mut w, vec![egui::Event::PointerMoved(mid + egui::vec2(-300.0, 0.0))]);
    let loading = drawn_honest(&w, &planes, "drag, worker held");
    assert_eq!(loading, 0, "pan discarded the complete presentation");
    assert!(w.whole_track.exact_camera().is_none());
    u.frame(&mut w, vec![press(mid + egui::vec2(-300.0, 0.0), egui::PointerButton::Primary, false)]);
    drawn_honest(&w, &planes, "drag released, worker held");
    release_worker(&mut w, &gate);
    until_exact(&mut w, &mut u, &planes, "drag");

    // Superseded: a zoom's job is in flight (held) when a second zoom asks
    // for another camera. The first job's columns land for spans the second
    // camera does not use; nothing of them is drawn as its data.
    let gate = hold_worker(&mut w);
    let wheel = vec![egui::Event::PointerMoved(mid),
        egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: egui::vec2(0.0, 40.0), modifiers: Default::default() }];
    u.frame(&mut w, wheel.clone());
    let first = asked(&w);
    assert!(w.whole_track.worker().busy(), "setup: the first zoom's job is not in flight");
    u.frame(&mut w, wheel);
    let second = asked(&w);
    assert_ne!(first, second, "setup: the second zoom asked for the same camera");
    release_worker(&mut w, &gate);
    let n = until_exact(&mut w, &mut u, &planes, "superseded");
    assert_eq!(asked(&w), second, "the view settled on the superseded camera");
    eprintln!("superseded: exact after {n} frame(s)");
}

/// Split and a window resize, with the worker held: the rings are new at the
/// new size, every column of both planes loads (none is the old size's
/// picture scaled), and both become exact together.
#[test]
fn split_and_resize_load_both_planes_honestly() {
    let d = Dir::new("wt_async_split");
    let track = tracks(&d, 1).remove(0);
    let mut w = open_window(&d.0.join("c"), true);
    w.on_play(&track, SR);
    assert!(settle(&mut w));
    let mut w = open_window(&d.0.join("c"), false);
    let mut u = Ui::new();
    w.on_play(&track, SR);
    assert!(settle(&mut w));
    w.whole_track.plane = Plane::Split;
    u.settle(&mut w);
    let pair = w.analyzer.pre_channels().resident().expect("setup: no channels").clone();
    let planes = [pair.left(), pair.right()];
    assert_eq!(drawn_honest(&w, &planes, "split settled"), 0);
    w.whole_track.sync_cells_for_test = Some(0);
    let gate = hold_worker(&mut w);
    let before = w.whole_track.drawn()[0].1;
    // A larger window.
    u.t += 1.0 / 60.0;
    let mut i = input(u.t, vec![]);
    i.screen_rect = Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(W + 300.0, H + 120.0)));
    let _ = u.ctx.run(i, |c| {
        egui::CentralPanel::default().show(c, |ui| w.whole_track_ui(ui));
    });
    let after = w.whole_track.drawn();
    assert_ne!(after[0].1, before, "setup: the rings were not resized");
    let loading = drawn_honest(&w, &planes, "resize, worker held");
    assert_eq!(loading, 2 * after[0].1, "a resized ring drew something before its columns were read");
    release_worker(&mut w, &gate);
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        u.t += 1.0 / 60.0;
        let mut i = input(u.t, vec![]);
        i.screen_rect = Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(W + 300.0, H + 120.0)));
        let _ = u.ctx.run(i, |c| {
            egui::CentralPanel::default().show(c, |ui| w.whole_track_ui(ui));
        });
        drawn_honest(&w, &planes, "resize, landing");
        if w.whole_track.exact_camera() == Some(asked(&w)) {
            break;
        }
        assert!(Instant::now() < deadline, "the resized Split never became exact");
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Stored codes at a realistic display width, without decoding audio or
/// running a transform. The install/record is the same shape the view
/// captures in production; only the source of the already-computed codes is
/// synthetic. No file is opened for the nominal track name.
fn memory_view_fixture(tag: &str, frames: usize, bars: usize, split: bool) -> (Dir, SpectrumWindow, Ui) {
    let d = Dir::new(tag);
    let mut w = open_window(&d.0.join("cache"), false);
    w.bar_count = bars;
    w.analyzer.set_bar_count(bars);
    w.analyzer.sample_rate = 48_000;
    w.analyzer.aslt_cfg.q_ratio = 2.0;
    w.analyzer.aslt_cfg.n_wavelets = 9;
    w.analyzer.aslt_cfg.max_window_s = 0.25;
    let settings = w.analyzer.effective_settings(Path::new("synthetic-view.wav"));
    let record = settings.record(48_000);
    let rate = wt::Geometry::parse(&record).unwrap().frame_rate();
    let plane = |seed: usize| {
        let codes = (0..frames).flat_map(|f| {
            (0..bars).map(move |b| {
                if (f * 19 + b * 7 + seed).is_multiple_of(137) {
                    4000
                } else {
                    ((f * 31 + b * 17 + seed * 211) % 3000) as u16
                }
            })
        }).collect();
        cache::PreFrames::from_v4_codes(codes, frames, bars)
    };
    let channels = if split {
        PreChannels::Resident(stereo::Stereo::new(plane(2), plane(3)).unwrap())
    } else {
        PreChannels::Missing
    };
    w.analyzer.set_pre_frames_with(plane(1), channels, None, rate);
    w.analyzer.pre_record = Some(record);
    w.whole_track.plane = if split { Plane::Split } else { Plane::Mix };
    let mut u = Ui::new();
    u.frame(&mut w, vec![]);
    let margin = W - w.whole_track.plot_for_test.unwrap().width();
    u.width = Some(1584.0 + margin);
    // This deliberately does not call SpectrumWindow::tick: the fixture has
    // no transport, cache reader, or analysis job to service. The real view
    // and its worker still run through whole_track_ui on every frame.
    settle_memory_view(&mut w, &mut u);
    assert_eq!(w.whole_track.drawn()[0].1, 1584, "setup: not the owner's raster width");
    assert_eq!(w.whole_track.snapshot().unwrap().geometry.data_bars, bars);
    (d, w, u)
}

fn settle_memory_view(w: &mut SpectrumWindow, u: &mut Ui) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        u.frame(w, vec![]);
        if w.whole_track.exact_camera().is_some() && !w.whole_track.worker().busy() {
            return;
        }
        assert!(Instant::now() < deadline, "synthetic stored-code view did not settle");
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn memory_view_planes(w: &SpectrumWindow) -> Vec<Arc<cache::PreFrames>> {
    let s = w.whole_track.snapshot().unwrap();
    match w.whole_track.plane {
        Plane::Mix => vec![s.mix.clone()],
        Plane::Split => {
            let (l, r) = s.lr.as_ref().unwrap();
            vec![l.clone(), r.clone()]
        }
        _ => panic!("this fixture only uses Mix or Split"),
    }
}

/// A slow reader is forced, not raced. Every wheel frame must keep the
/// exact uploaded old picture AND its axes/readout, accumulate new demand,
/// and upload nothing until all planes of the replacement are complete.
#[test]
fn a_wheel_burst_keeps_one_complete_picture_without_partial_uploads() {
    struct ReleaseOnDrop(Arc<AtomicBool>);
    impl Drop for ReleaseOnDrop {
        fn drop(&mut self) { self.0.store(false, Ordering::SeqCst); }
    }
    for split in [false, true] {
        let (_d, mut w, mut u) = memory_view_fixture("wt_coherent", 8192, 32, split);
        let planes = memory_view_planes(&w);
        let refs: Vec<_> = planes.iter().map(|p| &**p).collect();
        let rate = w.analyzer.pre_frame_rate;
        w.whole_track.playhead = 3600.0 / rate;
        w.whole_track.set_view_for_test(View { t0: 1800.0, t1: 5800.0, b0: 0.0, b1: 32.0 });
        settle_memory_view(&mut w, &mut u);
        assert_eq!(drawn_honest(&w, &refs, "before burst"), 0);
        let picture = w.whole_track.drawn();
        let shown = picture[0].0;
        let textures = w.whole_track.ring_textures_for_test();
        let uploaded = w.whole_track.usage.ring.upload_bytes;
        w.whole_track.sync_cells_for_test = Some(0);
        let gate = hold_worker(&mut w);
        let _release = ReleaseOnDrop(gate.clone());
        let plot = w.whole_track.plot_for_test.unwrap();
        let pos = egui::pos2(plot.left() + plot.width() * 0.3, plot.top() + plot.height() * 0.2);
        let mut previous_width = shown.t1 - shown.t0;
        for _ in 0..8 {
            u.frame(&mut w, vec![egui::Event::PointerMoved(pos),
                egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, -10.0), modifiers: Default::default() }]);
            let request = asked(&w);
            assert!(request.t1-request.t0 > previous_width, "wheel demand stopped accumulating behind the retained image");
            previous_width = request.t1-request.t0;
            assert_eq!(w.whole_track.drawn(), picture, "waiting changed visible codes or axes");
            assert_eq!(w.whole_track.ring_textures_for_test(), textures, "waiting changed visible texture mapping");
            assert_eq!(w.whole_track.usage.ring.upload_bytes, uploaded, "partial working data was uploaded over the held texture");
            assert!(w.whole_track.exact_camera().is_none(), "old display claimed to finish new demand");
            let (_, rw, rh, _) = &picture[0];
            let raster = w.whole_track.sub_for_test[0];
            let xy = (((pos.x-raster.left())/raster.width()* *rw as f32) as usize,
                ((pos.y-raster.top())/raster.height()* *rh as f32) as usize);
            let expected = crate::spectrum::wholetrack_ui::hover(&planes[0], &shown, (*rw,*rh), xy,
                w.whole_track.snapshot().unwrap(), if split { "Left" } else { "Mix" });
            assert_eq!(w.whole_track.readout_for_test(), expected, "hover used the unfinished camera");
        }
        assert!(!w.whole_track.follow_suspended);
        // Frequency rows can change on the CPU while a time request is
        // pending. This forces the partial-upload window; a held worker
        // alone leaves the texture clean and cannot expose that bug.
        let shift = egui::Modifiers { shift: true, ..Default::default() };
        u.frame_held(&mut w, vec![egui::Event::PointerMoved(pos),
            egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(0.0, 25.0), modifiers: shift }], shift);
        assert_eq!(w.whole_track.drawn(), picture);
        assert_eq!(w.whole_track.usage.ring.upload_bytes, uploaded, "frequency work overwrote the retained GPU texture");
        let desired = asked(&w);
        // Clicking the old picture must seek its old time, not the time
        // beneath an invisible replacement. A primary click does not pan.
        u.frame(&mut w, vec![press(pos, egui::PointerButton::Primary, true)]);
        u.frame(&mut w, vec![press(pos, egui::PointerButton::Primary, false)]);
        let raster = w.whole_track.sub_for_test[0];
        let hit = (shown.t0 + ((pos.x-raster.left())/raster.width()) as f64 * (shown.t1-shown.t0)).floor();
        let expected_seek = w.whole_track.snapshot().unwrap().geometry.frame_time(hit);
        assert_eq!(w.take_seek_request(), Some(expected_seek), "click sought the unfinished camera");
        release_worker(&mut w, &gate);
        until_exact(&mut w, &mut u, &refs, "coherent commit");
        assert_eq!(asked(&w), desired);
        assert_eq!(w.whole_track.painted_for_test, Some(desired));
        assert_eq!(drawn_honest(&w, &refs, "replacement committed"), 0);
        assert!(w.whole_track.usage.ring.upload_bytes > uploaded);
        // A held image must not survive a physical raster resize: that
        // would stretch it and break its pixel/data correspondence.
        let gate = hold_worker(&mut w);
        let _release_resize = ReleaseOnDrop(gate.clone());
        u.frame(&mut w, vec![egui::Event::PointerMoved(pos),
            egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(0.0, 10.0), modifiers: Default::default() }]);
        assert!(w.whole_track.exact_camera().is_none());
        u.width = Some(u.width.unwrap() + 64.0);
        u.frame(&mut w, vec![]);
        let new_columns: usize = w.whole_track.drawn().iter().map(|(_,rw,_,_)| rw).sum();
        assert_eq!(drawn_honest(&w, &refs, "resize cannot stretch retained image"), new_columns);
        release_worker(&mut w, &gate);
        until_exact(&mut w, &mut u, &refs, "resized commit");
        w.whole_track.close();
        assert!(!w.whole_track.holds_anything());
    }
}

/// A new analysis invalidates the held GPU presentation even when its old
/// background reader has not yet retired.
#[test]
fn a_retained_picture_is_dropped_when_the_analysis_changes() {
    let (_d, mut w, mut u) = memory_view_fixture("wt_presented_generation", 4096, 16, false);
    w.whole_track.follow = false;
    let pos = w.whole_track.plot_for_test.unwrap().center();
    let gate = hold_worker(&mut w);
    struct Release(Arc<AtomicBool>);
    impl Drop for Release { fn drop(&mut self) { self.0.store(false, Ordering::SeqCst); } }
    let _release = Release(gate.clone());
    w.whole_track.sync_cells_for_test = Some(0);
    u.frame(&mut w, vec![egui::Event::PointerMoved(pos),
        egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0.0, 30.0), modifiers: Default::default() }]);
    assert!(w.whole_track.exact_camera().is_none());
    assert!(w.whole_track.drawn()[0].3.iter().all(Option::is_some));
    let record = w.analyzer.pre_record.clone();
    let rate = w.analyzer.pre_frame_rate;
    let fresh = cache::PreFrames::from_v4_codes(vec![1234;4096*16],4096,16);
    w.analyzer.set_pre_frames_with(fresh, PreChannels::Missing, None, rate);
    w.analyzer.pre_record = record;
    u.frame(&mut w, vec![]);
    assert!(w.whole_track.drawn()[0].3.iter().all(Option::is_none), "old analysis survived as a retained picture");
    release_worker(&mut w, &gate);
    let plane = w.analyzer.pre_frames.clone();
    until_exact(&mut w, &mut u, &[&plane], "new generation");
    assert!(w.whole_track.drawn()[0].3.iter().flatten().flatten().all(|&c| c == 1234));
    w.whole_track.close();
    assert!(!w.whole_track.holds_anything());
}

/// Short, headless CPU measurement of the real view (not a native/GPU
/// presentation benchmark). Uses already-computed stored codes, so
/// decoding, ASLT and cache I/O cannot hide the interaction cost.
#[test]
#[ignore = "component CPU probe: run alone in release"]
fn wholetrack_zoom_component_probe() {
    #[cfg(windows)]
    fn cpu() -> f64 {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> isize;
            fn GetProcessTimes(h: isize, c: *mut u64, e: *mut u64, k: *mut u64, u: *mut u64) -> i32;
        }
        let (mut c, mut e, mut k, mut u) = (0, 0, 0, 0);
        assert_ne!(unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) }, 0);
        (k + u) as f64 / 1e7
    }
    #[cfg(not(windows))]
    fn cpu() -> f64 { 0.0 }

    let bars = std::env::var("MOOSIK_ZOOM_PROBE_BARS").ok()
        .map(|s| s.parse::<usize>().unwrap()).unwrap_or(1024);
    assert!([1024, 2048].contains(&bars));
    let wide = std::env::var_os("MOOSIK_ZOOM_PROBE_WIDE").is_some();
    let (_d, mut w, mut u) = memory_view_fixture("wt_zoom_cpu", if wide { 65_536 } else { 16_384 }, bars, false);
    let rate = w.analyzer.pre_frame_rate;
    let cores = std::thread::available_parallelism().unwrap().get() as f64;
    let ds: &[f64] = if wide {
        &[8.5, 12.5, 16.5, 32.5]
    } else { &[0.5, 1.5, 3.9, 6.7] };
    for &d in ds {
        let mut ph_frame = 8192.0;
        w.whole_track.playhead = ph_frame / rate;
        w.whole_track.set_view_for_test(View { t0: 0.0, t1: 1584.0 * d, b0: 0.0, b1: bars as f64 });
        settle_memory_view(&mut w, &mut u);
        let mid = w.whole_track.plot_for_test.unwrap().center();
        let before = w.whole_track.usage;
        let mut times = Vec::new();
        let (c0, start) = (cpu(), Instant::now());
        for i in 0..360 {
            let due = start + Duration::from_secs_f64(i as f64 / 180.0);
            if let Some(wait) = due.checked_duration_since(Instant::now()) { std::thread::sleep(wait); }
            ph_frame += rate / 180.0;
            w.whole_track.playhead = ph_frame / rate;
            let mut events = vec![egui::Event::PointerMoved(mid)];
            if i % 18 == 0 {
                let dy = if (i / 18) % 6 < 3 { 40.0 } else { -40.0 };
                events.push(egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, dy), modifiers: Default::default() });
            }
            let t = Instant::now();
            u.frame(&mut w, events);
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let wall = start.elapsed().as_secs_f64();
        let machine = 100.0 * (cpu() - c0) / wall / cores;
        times.sort_by(f64::total_cmp);
        let after = w.whole_track.usage;
        let waiting = after.catching_up_frames - before.catching_up_frames;
        let retained = after.retained_frames - before.retained_frames;
        println!("{{\"bars\":{bars},\"d\":{d},\"frames\":360,\"wall_s\":{wall:.3},\"cpu_machine_pct\":{machine:.2},\"ui_p95_ms\":{:.3},\"loading_frames\":{},\"retained_frames\":{retained},\"worker_requests\":{}}}",
            times[342], waiting - retained, after.requests - before.requests);
    }
}

/// Magnified time windows still need to work at the actual width, including
/// the extra 2048-bar Split stress case: counting every duplicated pixel's
/// one-frame row as a scan
/// otherwise sends a tiny visible time range to the worker. A wheel step
/// and the advancing follow camera must draw their own exact codes without
/// starting any new asynchronous job.
#[test]
fn magnified_wheel_and_follow_at_owner_width_never_load_mix_or_split() {
    for bars in [1024usize, 2048] {
        for split in [false, true] {
            let (_d, mut w, mut u) = memory_view_fixture("wt_short_window", 8192, bars, split);
            let planes = memory_view_planes(&w);
            let refs: Vec<_> = planes.iter().map(|p| &**p).collect();
            let rate = w.analyzer.pre_frame_rate;
            let mut ph_frame = 3600.0;
            let widths: &[f64] = if split { &[0.5, 1.0] } else { &[0.5, 1.0, 1.1, 2.1] };
            for &d in widths {
                w.whole_track.playhead = ph_frame / rate;
                w.whole_track.set_view_for_test(View { t0: 0.0, t1: 1584.0 * d, b0: 0.0, b1: bars as f64 });
                settle_memory_view(&mut w, &mut u);
                let started = w.whole_track.worker().started;
                let mid = w.whole_track.plot_for_test.unwrap().center();
                // Zoom in first, including repeated nearest-frame spans in
                // the deep case and a ready short-query level otherwise.
                for (i, dy) in [12.0f32, -6.0].into_iter().enumerate() {
                    ph_frame += 1.0;
                    w.whole_track.playhead = ph_frame / rate;
                    u.frame(&mut w, vec![egui::Event::PointerMoved(mid),
                        egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: egui::vec2(0.0, dy), modifiers: Default::default() }]);
                    let v = asked(&w);
                    let what = format!("bars {bars}, split {split}, initial d {d}, wheel {i}");
                    assert_eq!(drawn_honest(&w, &refs, &what), 0, "{what}: a small time window loaded");
                    assert_eq!(w.whole_track.exact_camera(), Some(v), "{what}: another camera was painted");
                    assert!(!w.whole_track.worker().busy(), "{what}: work queued for a magnified time window");
                    assert_eq!(w.whole_track.worker().started, started, "{what}: a worker started");
                    assert!(!w.whole_track.follow_suspended, "{what}: following was set aside");
                    assert!(((ph_frame - v.t0) / (v.t1 - v.t0) - 1.0 / 3.0).abs() < 0.002, "{what}: the camera stopped following");
                }
            }
        }
    }
}

/// 5.1..7.3 geometric frames/pixel includes integer intersection lengths
/// 6..9: an octave selected from a median can drop the short spans, and a
/// two-block k=2 query drops the 9-frame spans. Warm the shared spatial range
/// at 6.7 first, then cross both cases with actual wheel input while following.
#[test]
fn wheel_crossing_eight_and_nine_frame_spans_uses_the_ready_level_without_loading() {
    let (_d, mut w, mut u) = memory_view_fixture("wt_eight_nine", 16_384, 64, false);
    let planes = memory_view_planes(&w);
    let refs: Vec<_> = planes.iter().map(|p| &**p).collect();
    let rate = w.analyzer.pre_frame_rate;
    let mut ph_frame = 8192.0;
    w.whole_track.playhead = ph_frame / rate;
    w.whole_track.set_view_for_test(View { t0: 0.0, t1: 1584.0 * 6.7, b0: 0.0, b1: 64.0 });
    // One scan row per pixel: ready 2/3-block queries fit, while an
    // unsupported interval's raw/pyramid scan cannot hide this regression.
    w.whole_track.sync_cells_for_test = Some(1584 * 64);
    settle_memory_view(&mut w, &mut u);
    let started = w.whole_track.worker().started;
    let mid = w.whole_track.plot_for_test.unwrap().center();
    for d in [5.1f64, 7.3, 6.7, 7.3, 5.1] {
        let before = asked(&w);
        let current = (before.t1 - before.t0) / 1584.0;
        let dy = -((d / current).ln() / 0.003) as f32;
        ph_frame += 1.0;
        w.whole_track.playhead = ph_frame / rate;
        u.frame(&mut w, vec![egui::Event::PointerMoved(mid),
            egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: egui::vec2(0.0, dy), modifiers: Default::default() }]);
        let v = asked(&w);
        assert!(((v.t1 - v.t0) / 1584.0 - d).abs() < 1e-5, "setup: the wheel did not reach d {d}");
        assert_eq!(drawn_honest(&w, &refs, &format!("d {d}")), 0, "d {d}: the ready level's coverage was lost");
        assert_eq!(w.whole_track.exact_camera(), Some(v));
        assert_eq!(w.whole_track.worker().started, started, "d {d}: the level was rebuilt");
        assert!(!w.whole_track.worker().busy(), "d {d}: asynchronous work queued");
        assert!(!w.whole_track.follow_suspended);
    }
}

/// The owner's 1.5.4 report: rolling the wheel to zoom time while following
/// flashed black with "loading…". A zoom inside the installed level's octave
/// is read on the UI thread through that level (two rows a column, counted at
/// their measured cost, see `wt::held_cost`), so every frame of a run of
/// wheel steps while the playhead advances is the asked camera, exact, with
/// nothing loading, and following carries on. The budget is set to one scan
/// row a column: any column read frame by frame still goes to the worker
/// (as `async_camera_changes_draw_loading_never_another_cameras_picture`
/// checks), while the level's refill fits — as at the owner's geometry, where
/// it is 0.8 Mi of the 2 Mi cells.
#[test]
fn wheel_zoom_while_following_inside_an_octave_never_loads() {
    let d = Dir::new("wt_wheel_follow");
    // Ten seconds: enough frames for a level-sized column at a plot this
    // harness can draw.
    let track = d.0.join("long.wav");
    write_wav_secs(&track, &[300.0, 2100.0], 7, 0.05, 10);
    let mut w = open_window(&d.0.join("c"), false);
    let mut u = Ui::new();
    w.on_play(&track, SR);
    assert!(settle(&mut w));
    u.settle(&mut w);
    let plane = (*w.analyzer.pre_frames).clone();
    let planes = [&plane];
    let rate = w.analyzer.pre_frame_rate;
    let frames = plane.len();
    // A plot about six frames a column wide (a level of octave 4..8).
    let margin = W - w.whole_track.plot_for_test.unwrap().width();
    let cols = frames as f32 * 0.8 / 6.0;
    assert!(cols >= 24.0, "setup: the fixture is too short ({frames} frames)");
    u.width = Some(cols + margin);
    let full = w.whole_track.view().unwrap();
    w.whole_track.set_view_for_test(View { t0: 0.0, t1: full.t1 * 0.8, ..full });
    let mut ph = 3.5; // past a third of the window, short of its end
    w.tick(ph, true);
    u.settle(&mut w);
    assert!(w.whole_track.follow && !w.whole_track.follow_suspended, "setup: not following");
    let mid = w.whole_track.plot_for_test.unwrap().center();
    let wheel = |dy: f32| vec![egui::Event::PointerMoved(mid),
        egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: egui::vec2(0.0, dy), modifiers: Default::default() }];
    let rw = w.whole_track.drawn()[0].1;
    let db = w.whole_track.snapshot().unwrap().geometry.data_bars;
    w.whole_track.sync_cells_for_test = Some(rw * db);
    // One step first: read frame by frame it is over the budget, so the
    // worker reads it and builds the level for this octave.
    ph += 1.0 / 60.0;
    w.tick(ph, true);
    u.frame(&mut w, wheel(10.0));
    assert_eq!(drawn_honest(&w, &planes, "first wheel step"), 0);
    assert!(w.whole_track.exact_camera().is_none(), "setup: the first step was not left to the worker");
    until_exact(&mut w, &mut u, &planes, "first wheel step");
    let per_col = |v: View| (v.t1 - v.t0) / rw as f64;
    assert!((4.0..8.0).contains(&per_col(asked(&w))), "setup: {:.2} frames a column ({rw} columns, {frames} frames)", per_col(asked(&w)));
    let at = |v: View, ph: f64| (ph * rate - v.t0) / (v.t1 - v.t0);
    let mut last = asked(&w);
    for (i, dy) in [10.0, -10.0, 10.0, 10.0, -10.0, -10.0, 10.0, -10.0].into_iter().enumerate() {
        ph += 1.0 / 60.0;
        w.tick(ph, true);
        u.frame(&mut w, wheel(dy));
        let v = asked(&w);
        assert!((v.t1 - v.t0 - (last.t1 - last.t0)).abs() > 1e-9, "setup: step {i} did not zoom");
        assert!((4.0..8.0).contains(&per_col(v)), "setup: step {i} left the octave ({:.2})", per_col(v));
        assert_eq!(drawn_honest(&w, &planes, "wheel while following"), 0, "step {i}: columns drawn as loading");
        assert_eq!(w.whole_track.exact_camera(), Some(v), "step {i}: the asked camera was not drawn exactly");
        assert!(!w.whole_track.follow_suspended, "step {i}: the wheel set following aside");
        assert!((at(v, ph) - 1.0 / 3.0).abs() < 0.02, "step {i}: the playhead left its place ({:.3})", at(v, ph));
        last = v;
    }
}

/// Code-exact is not pixel-exact: the meshes egui is given place each camera
/// column on its own pixel, sampling the texture slot that holds it. Checked
/// on the actual paint output after the ring has wrapped (a scrolled view),
/// at both edges of every textured rectangle.
#[test]
fn the_ring_texture_maps_each_camera_column_to_its_own_pixel() {
    let d = Dir::new("wt_texmap");
    let track = tracks(&d, 1).remove(0);
    let mut w = open_window(&d.0.join("c"), false);
    let mut u = Ui::new();
    w.on_play(&track, SR);
    assert!(settle(&mut w));
    u.settle(&mut w);
    let full = w.whole_track.view().unwrap();
    w.whole_track.set_view_for_test(View { t0: 0.0, t1: full.t1 * 0.25, b0: full.b1 * 0.113, b1: full.b1 * 0.61 });
    // Follow mid-track, a few frames apart, until the ring has wrapped
    // (its base is not a multiple of its width).
    let secs = full.t1 / w.analyzer.pre_frame_rate;
    for i in 0..240 {
        w.tick(secs * 0.5 + i as f64 * secs / 4000.0, true);
        u.frame(&mut w, vec![]);
        if w.whole_track.ring_textures_for_test().first().is_some_and(|t| t.1.rem_euclid(t.2 as i64) != 0 && t.4 != 0)
            && w.whole_track.exact_camera().is_some()
        {
            break;
        }
    }
    let cam = w.whole_track.exact_camera().expect("setup: not exact");
    let tex = w.whole_track.ring_textures_for_test();
    let (id, base, rw, rh, top) = tex[0];
    assert_ne!(base.rem_euclid(rw as i64), 0, "setup: the ring has not wrapped in time");
    assert_ne!(top, 0, "setup: the ring has not wrapped in frequency");
    let plot = w.whole_track.plot_for_test.unwrap();
    u.t += 1.0 / 60.0;
    let out = u.ctx.run(input(u.t, vec![]), |c| {
        egui::CentralPanel::default().show(c, |ui| w.whole_track_ui(ui));
    });
    assert_eq!(w.whole_track.exact_camera(), Some(cam), "setup: the camera moved");
    let mut covered = 0.0;
    for cs in &out.shapes {
        let egui::epaint::Shape::Mesh(m) = &cs.shape else { continue };
        if m.texture_id != id {
            continue;
        }
        let (lo, hi) = m.vertices.iter().fold((egui::pos2(f32::MAX, f32::MAX), egui::pos2(f32::MIN, f32::MIN)), |(a, b), v| (a.min(v.pos), b.max(v.pos)));
        let (ulo, uhi) = m.vertices.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(v.uv.x), b.max(v.uv.x)));
        let (vlo, vhi) = m.vertices.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(v.uv.y), b.max(v.uv.y)));
        // On the pixel grid (pixels_per_point is 1 here).
        for c in [lo.x, hi.x, lo.y, hi.y] {
            assert_eq!(c, c.round(), "a textured rectangle edge lies between pixels");
        }
        // Rows: one texel per pixel, the top and bottom pixel rows sampling
        // the texture rows their camera rows are kept in.
        let py = hi.y - lo.y;
        assert!((py - (vhi - vlo) * rh as f32).abs() < 1e-3, "{py} pixel rows for {} texel rows", (vhi - vlo) * rh as f32);
        for (y, v_at) in [(lo.y + 0.5, vlo + 0.5 / rh as f32), (hi.y - 0.5, vhi - 0.5 / rh as f32)] {
            let row = ((y - plot.top()) as usize).min(rh - 1);
            assert_eq!((v_at * rh as f32) as usize, (top + row) % rh, "pixel row {y} samples the wrong texture row");
        }
        // One texel per pixel.
        let px = hi.x - lo.x;
        assert!((px - (uhi - ulo) * rw as f32).abs() < 1e-3, "{px} pixels for {} texels", (uhi - ulo) * rw as f32);
        // The first and last pixel sample the slots of their camera columns.
        for (x, u_at) in [(lo.x + 0.5, ulo + 0.5 / rw as f32), (hi.x - 0.5, uhi - 0.5 / rw as f32)] {
            let col = ((x - plot.left()) as usize).min(rw - 1);
            let slot = (u_at * rw as f32) as i64;
            assert_eq!(slot, (base + col as i64).rem_euclid(rw as i64), "pixel {x} samples slot {slot}");
        }
        covered += px * py;
    }
    assert_eq!(covered as usize, rw * rh, "the textured rectangles do not cover the plot exactly once");
}

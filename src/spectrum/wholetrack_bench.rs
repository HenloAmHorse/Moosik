//! Headless component benchmark of the whole-track view (ignored; run by
//! hand with `--ignored --nocapture` in release). A synthetic long analysis
//! at the owner's geometry (1024 bars, 180 frames/s) is installed directly,
//! and the view is driven at a 60 Hz cadence through the same headless egui
//! frames the integration tests use, along matched camera trajectories.
//!
//! What it measures is this process's CPU time (all threads: the UI pass and
//! the raster worker) per pattern, plus the view's own counters. It does NOT
//! include tessellation, the GPU upload or presenting a frame: those are the
//! native measurement's job. It is a component cost model, labelled as such.
use super::*;
use crate::spectrum::wholetrack::View;
use crate::spectrum::wholetrack_ui::Plane;

const BW: f32 = 1600.0;
const BH: f32 = 900.0;
const BENCH_SR: u32 = 48_000;
const BENCH_BARS: usize = 1024;

fn input_at(t: f64, events: Vec<egui::Event>) -> egui::RawInput {
    // Shift is held for a frame whose wheel event says so: egui reads the
    // frame's modifiers, as the platform layer reports them.
    let shift = events.iter().any(|e| matches!(e, egui::Event::MouseWheel { modifiers, .. } if modifiers.shift));
    egui::RawInput {
        modifiers: egui::Modifiers { shift, ..Default::default() },
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(BW, BH))),
        max_texture_side: Some(8192),
        time: Some(t),
        events,
        ..Default::default()
    }
}

#[cfg(windows)]
fn cpu_seconds() -> f64 {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetProcessTimes(p: isize, c: *mut u64, e: *mut u64, k: *mut u64, u: *mut u64) -> i32;
    }
    let (mut c, mut e, mut k, mut u) = (0, 0, 0, 0);
    unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) };
    (k + u) as f64 / 1e7
}
#[cfg(not(windows))]
fn cpu_seconds() -> f64 {
    0.0
}

/// This thread's CPU time (the UI pass; the worker runs on its own threads).
#[cfg(windows)]
fn thread_seconds() -> f64 {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> isize;
        fn GetThreadTimes(t: isize, c: *mut u64, e: *mut u64, k: *mut u64, u: *mut u64) -> i32;
    }
    let (mut c, mut e, mut k, mut u) = (0, 0, 0, 0);
    unsafe { GetThreadTimes(GetCurrentThread(), &mut c, &mut e, &mut k, &mut u) };
    (k + u) as f64 / 1e7
}
#[cfg(not(windows))]
fn thread_seconds() -> f64 {
    0.0
}

/// A plane with structure at every scale: tones, sweeps, transients, noise.
fn plane(frames: usize, seed: u32) -> cache::PreFrames {
    let mut x = seed.max(1);
    let mut codes = Vec::with_capacity(frames * BENCH_BARS);
    for f in 0..frames {
        for b in 0..BENCH_BARS {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let mut c = (x % 900) as u16;
            if b % 97 == (f / 500) % 97 { c = c.max(3000); }
            if (f + b * 7) % 4001 == 0 { c = 4095; }
            if b == (f / 40) % BENCH_BARS { c = c.max(2500); }
            codes.push(c.min(4095));
        }
    }
    cache::PreFrames::from_v4_codes(codes, frames, BENCH_BARS)
}

/// A window showing a synthetic analysis of `secs` seconds, Mix and channels.
fn bench_window(d: &Dir, secs: f64) -> SpectrumWindow {
    let track = d.0.join("bench.wav");
    std::fs::write(&track, b"not audio; the analysis is installed directly").unwrap();
    let mut w = window(&d.0.join("c"), 1.0);
    w.open = true;
    w.whole_track.open = true;
    w.bar_count = BENCH_BARS;
    w.analyzer.bar_count = BENCH_BARS;
    w.analyzer.pre_fps = 180.0;
    w.analyzer.sample_rate = BENCH_SR;
    let record = w.analyzer.effective_settings(&track).record(BENCH_SR);
    let g = crate::spectrum::wholetrack::Geometry::parse(&record).expect("bench record");
    let frames = (secs * g.frame_rate()) as usize;
    let mix = plane(frames, 7);
    let pair = stereo::Stereo::new(plane(frames, 11), plane(frames, 13)).unwrap();
    w.analyzer.set_pre_frames_with(mix, PreChannels::Resident(pair), None, g.frame_rate());
    w.analyzer.pre_record = Some(record);
    w.current_path = Some(track);
    follow_off(&mut w);
    w
}

/// The renderer's own following is off: every pattern moves the camera
/// itself. (A build without following defines this as nothing.)
fn follow_off(w: &mut SpectrumWindow) {
    w.whole_track.follow = false;
}

struct Run {
    name: &'static str,
    cpu: f64,
    wall: f64,
    frames: usize,
    ui_cpu: f64,
    requests: u64,
    uploads: u64,
}

/// Drive `n` frames at 60 Hz; `step(i)` returns the events for frame `i`
/// and may move the view or the playhead first.
fn drive(w: &mut SpectrumWindow, ctx: &egui::Context, t: &std::cell::Cell<f64>, n: usize, mut step: impl FnMut(&mut SpectrumWindow, usize) -> Vec<egui::Event>) -> (f64, f64, f64) {
    let (c0, w0, u0) = (cpu_seconds(), Instant::now(), thread_seconds());
    for i in 0..n {
        let due = w0 + Duration::from_secs_f64(i as f64 / 60.0);
        if let Some(s) = due.checked_duration_since(Instant::now()) {
            std::thread::sleep(s);
        }
        let ev = step(w, i);
        t.set(t.get() + 1.0 / 60.0);
        let _ = ctx.run(input_at(t.get(), ev), |c| {
            egui::CentralPanel::default().show(c, |ui| w.whole_track_ui(ui));
        });
    }
    (cpu_seconds() - c0, w0.elapsed().as_secs_f64(), thread_seconds() - u0)
}

#[test]
#[ignore = "benchmark: run by hand in release"]
fn whole_track_interaction_component_cost() {
    let secs: f64 = std::env::var("MOOSIK_BENCH_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(300.0);
    let n: usize = std::env::var("MOOSIK_BENCH_FRAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(1800);
    let d = Dir::new("wt_bench");
    let ctx = egui::Context::default();
    let t = std::cell::Cell::new(0.0);
    let mut out = Vec::new();
    for plane in [Plane::Mix, Plane::Split] {
        for zoom in [1.0f64, 0.25] {
            // A fresh window each time: a playing tick releases channels
            // nothing on screen needs, and Split needs them resident.
            let mut w = bench_window(&d, secs);
            w.whole_track.plane = plane;
            // Settle on the starting view.
            let frames = w.analyzer.pre_frames.len() as f64;
            let span = frames * zoom;
            let start = View { t0: (frames - span) * 0.3, t1: (frames - span) * 0.3 + span, b0: 0.0, b1: BENCH_BARS as f64 };
            w.whole_track.set_view_for_test(start);
            drive(&mut w, &ctx, &t, 120, |_, _| vec![]);
            let plot = w.whole_track.plot_for_test.expect("no plot");
            let mid = plot.center();
            let tag = |p: &str| -> &'static str { Box::leak(format!("{p} {:?} zoom={zoom}", plane).into_boxed_str()) };
            let mut pattern = |name: &'static str, w: &mut SpectrumWindow, step: &mut dyn FnMut(&mut SpectrumWindow, usize) -> Vec<egui::Event>| {
                if std::env::var("MOOSIK_BENCH_ONLY").is_ok_and(|o| !name.contains(o.as_str())) {
                    return;
                }
                w.whole_track.set_view_for_test(start);
                drive(w, &ctx, &t, 60, |_, _| vec![]);
                let (r0, u0) = (w.whole_track.usage.requests, w.whole_track.usage.ring.upload_bytes);
                let (cpu, wall, ui_cpu) = drive(w, &ctx, &t, n, |w, i| step(w, i));
                out.push(Run { name, cpu, wall, frames: n, ui_cpu, requests: w.whole_track.usage.requests - r0, uploads: w.whole_track.usage.ring.upload_bytes - u0 });
            };
            // Playing, nothing touched.
            let mut ph = 0.0;
            pattern(tag("stationary-playing"), &mut w, &mut |w, _| { ph += 1.0 / 60.0; w.tick(ph, true); vec![] });
            // Pointer moving over the plot, no camera change.
            pattern(tag("hover"), &mut w, &mut |_, i| {
                let x = mid.x + ((i % 240) as f32 - 120.0) * 3.0;
                vec![egui::Event::PointerMoved(egui::pos2(x, mid.y))]
            });
            // A primary drag back and forth, 6 points per frame.
            pattern(tag("drag"), &mut w, &mut |_, i| {
                let k = i % 240;
                let x = mid.x + if k < 120 { k as f32 * 6.0 - 360.0 } else { 360.0 - (k - 120) as f32 * 6.0 };
                let p = egui::pos2(x, mid.y);
                match i {
                    0 => vec![egui::Event::PointerMoved(p), egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: true, modifiers: Default::default() }],
                    _ => vec![egui::Event::PointerMoved(p)],
                }
            });
            w.whole_track.set_view_for_test(start);
            drive(&mut w, &ctx, &t, 2, |_, _| vec![egui::Event::PointerButton { pos: mid, button: egui::PointerButton::Primary, pressed: false, modifiers: Default::default() }]);
            // From here the camera is set directly, frame by frame, along the
            // same trajectory in every build (the input path is not compared).
            let px = (start.t1 - start.t0) / plot.width() as f64;
            // A pan back and forth, 6 points per frame.
            pattern(tag("pan-trajectory"), &mut w, &mut |w, i| {
                let k = (i % 240) as f64;
                let off = if k < 120.0 { k * 6.0 } else { (240.0 - k) * 6.0 } * px;
                w.whole_track.set_view_for_test(View { t0: start.t0 + off, t1: start.t1 + off, ..start });
                vec![egui::Event::PointerMoved(mid)]
            });
            // Time zoom about the centre: a 5 % step every third frame, in for
            // 2 s, out for 2 s (wheel-notch-like).
            let mut v = start;
            pattern(tag("zoom-trajectory"), &mut w, &mut |w, i| {
                if i % 3 == 0 {
                    let k = if (i / 120) % 2 == 0 { 0.95 } else { 1.0 / 0.95 };
                    let c = (v.t0 + v.t1) / 2.0;
                    let h = (v.t1 - v.t0) * k / 2.0;
                    v = View { t0: c - h, t1: c + h, ..v };
                    w.whole_track.set_view_for_test(v);
                }
                vec![egui::Event::PointerMoved(mid)]
            });
            // Frequency zoom about the centre, the same way.
            let mut v = start;
            pattern(tag("freqzoom-trajectory"), &mut w, &mut |w, i| {
                if i % 3 == 0 {
                    let k = if (i / 120) % 2 == 0 { 0.95 } else { 1.0 / 0.95 };
                    let c = (v.b0 + v.b1) / 2.0;
                    let h = (v.b1 - v.b0) * k / 2.0;
                    v = View { b0: c - h, b1: c + h, ..v };
                    w.whole_track.set_view_for_test(v);
                }
                vec![egui::Event::PointerMoved(mid)]
            });
            // Continuous follow: the camera moves with the playhead every frame.
            let mut ph = 0.0;
            pattern(tag("follow-trajectory"), &mut w, &mut |w, _| {
                ph += 1.0 / 60.0;
                w.tick(start.t0 / 180.0 + ph, true);
                let dt = ph * w.analyzer.pre_frame_rate;
                w.whole_track.set_view_for_test(View { t0: start.t0 + dt, t1: start.t1 + dt, ..start });
                vec![]
            });
        }
    }
    let path = std::env::var("MOOSIK_BENCH_OUT").ok();
    let mut text = String::new();
    for r in &out {
        let line = format!(
            "{{\"pattern\":\"{}\",\"cpu_s\":{:.4},\"wall_s\":{:.3},\"frames\":{},\"pct_one_core\":{:.2},\"pct_machine8\":{:.2},\"ui_thread_pct_one_core\":{:.2},\"requests\":{},\"upload_bytes\":{}}}",
            r.name, r.cpu, r.wall, r.frames, 100.0 * r.cpu / r.wall, 100.0 * r.cpu / r.wall / 8.0, 100.0 * r.ui_cpu / r.wall, r.requests, r.uploads
        );
        println!("{line}");
        text.push_str(&line);
        text.push('\n');
    }
    if let Some(p) = path {
        std::fs::write(p, text).unwrap();
    }
}

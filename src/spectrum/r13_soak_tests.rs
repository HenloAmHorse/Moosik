//! Round 13 sustained lifecycle/resource soak (ignored; run alone under the
//! fail-closed runner with an isolated profile). Headless `SpectrumWindow`
//! pre-process lifecycle only: this is not GUI, audio or rendering evidence.
//!
//! A seeded schedule over a small generated corpus (44.1, 48 and 96 kHz PCM
//! WAV, and a DSD64 DSF) at the owner's analysis descriptor: fresh analyses,
//! cache reuse, cancellation by track switch or stop at random points,
//! seeks, pause/resume, channel-view changes and a settings-generation change.
//! After every iteration the window is stopped and must go quiet; then:
//! - every completed install of a (track, generation) has exactly the codes of
//!   its first completion (Mix, and Left/Right when resident) — no stale or
//!   partial installation;
//! - no temporary cache file remains;
//! - cancellation went quiet within the bound.
//!
//! Resources per iteration: private commit, working set, handle count, thread
//! count, cache directory bytes. One JSON line per iteration to
//! `MOOSIK_R13_SOAK_OUT`; stops at `MOOSIK_R13_SOAK_HOURS` or at the first
//! failure (written, then panics).
//!
//! **Scope (R13 closeout).** The schedule's cache directory persists, so once
//! the four fixtures at both bar counts are cached (iteration 44 in R13's
//! ten-hour run) almost every later "complete" is a cache reuse: that run is a
//! long headless cache-replay/lifecycle/resource soak with 11 producer starts,
//! 8 fresh completions and 3 aborts. Fresh-work coverage is gated by
//! `r13x_fresh_tests::r13x_fresh_work_gate`. Since the closeout this soak also
//! counts producer starts, fresh completions, aborts and reuses separately,
//! fails a run-to-completion that installs nothing it can account for, fails a
//! requested channel view that never becomes resident, checks that what is
//! installed is the requested track at the selected settings, and includes
//! sidecar-reader retirement in quiescence (`r13x_fresh_tests`' checks).

use super::*;
use super::r13x_fresh_tests::{outcome_count, outcomes_since, quiesce, require_resident, spawned, verify_fresh, verify_installed};
use std::time::{Duration, Instant};

#[repr(C)]
#[derive(Default)]
struct Pmc { cb: u32, page_faults: u32, peak_ws: usize, ws: usize, qppp: usize, qpp: usize, qpnpp: usize, qnpp: usize, pagefile: usize, peak_pagefile: usize, private: usize }
#[repr(C)]
struct ThreadEntry32 { size: u32, usage: u32, tid: u32, owner_pid: u32, base_pri: i32, delta_pri: i32, flags: u32 }
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentProcess() -> isize;
    fn GetCurrentProcessId() -> u32;
    fn K32GetProcessMemoryInfo(process: isize, counters: *mut Pmc, cb: u32) -> i32;
    fn GetProcessHandleCount(process: isize, count: *mut u32) -> i32;
    fn CreateToolhelp32Snapshot(flags: u32, pid: u32) -> isize;
    fn Thread32First(snap: isize, e: *mut ThreadEntry32) -> i32;
    fn Thread32Next(snap: isize, e: *mut ThreadEntry32) -> i32;
    fn CloseHandle(h: isize) -> i32;
}

/// (private commit, working set, handles, threads).
pub(super) fn resources() -> (u64, u64, u32, u32) {
    unsafe {
        let mut c = Pmc { cb: std::mem::size_of::<Pmc>() as u32, ..Default::default() };
        assert!(K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) != 0);
        let mut h = 0u32;
        assert!(GetProcessHandleCount(GetCurrentProcess(), &mut h) != 0);
        let snap = CreateToolhelp32Snapshot(4, 0);
        let me = GetCurrentProcessId();
        let mut n = 0u32;
        let mut e = ThreadEntry32 { size: std::mem::size_of::<ThreadEntry32>() as u32, usage: 0, tid: 0, owner_pid: 0, base_pri: 0, delta_pri: 0, flags: 0 };
        if Thread32First(snap, &mut e) != 0 {
            loop {
                if e.owner_pid == me { n += 1; }
                if Thread32Next(snap, &mut e) == 0 { break; }
            }
        }
        CloseHandle(snap);
        (c.private as u64, c.ws as u64, h, n)
    }
}

pub(super) fn write_wav(path: &Path, sr: u32, secs: f64, f_l: f32, f_r: f32, seed: u64) {
    let n = (sr as f64 * secs) as usize;
    let mut s = seed;
    let mut data = Vec::with_capacity(n * 4);
    for i in 0..n {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        let t = i as f32 / sr as f32;
        let noise = ((s >> 52) as f32 / 4096.0 - 0.5) * 0.02;
        let l = ((std::f32::consts::TAU * f_l * t).sin() * 0.4 + noise) * 32767.0;
        let r = ((std::f32::consts::TAU * f_r * t).sin() * 0.3 + if i % (sr as usize / 3) < 2 { 0.5 } else { 0.0 }) * 32767.0;
        data.extend((l as i16).to_le_bytes());
        data.extend((r as i16).to_le_bytes());
    }
    let mut f = Vec::new();
    f.extend(b"RIFF"); f.extend(((36 + data.len()) as u32).to_le_bytes()); f.extend(b"WAVEfmt ");
    f.extend(16u32.to_le_bytes()); f.extend(1u16.to_le_bytes()); f.extend(2u16.to_le_bytes());
    f.extend(sr.to_le_bytes()); f.extend((sr * 4).to_le_bytes()); f.extend(4u16.to_le_bytes()); f.extend(16u16.to_le_bytes());
    f.extend(b"data"); f.extend((data.len() as u32).to_le_bytes()); f.extend(data);
    std::fs::write(path, f).unwrap();
}

/// Installed codes when only Mix was asked for (Left/Right 0 unless resident).
fn installed_codes_mix(w: &SpectrumWindow) -> (usize, u64, u64, u64) {
    super::r13x_fresh_tests::installed_codes(w)
}

fn hash_frames(p: &cache::PreFrames) -> (usize, u64) {
    let mut h = 0xcbf29ce484222325u64;
    for i in 0..p.len() {
        for v in p.row(i).unwrap_or(&[]) {
            h = (h ^ *v as u64).wrapping_mul(0x100000001b3);
        }
    }
    (p.len(), h)
}

fn dir_bytes(d: &Path) -> (u64, usize) {
    let (mut b, mut tmp) = (0u64, 0usize);
    if let Ok(rd) = std::fs::read_dir(d) {
        for e in rd.flatten() {
            if let Ok(m) = e.metadata() { b += m.len(); }
            if e.file_name().to_string_lossy().contains(".tmp-") { tmp += 1; }
        }
    }
    (b, tmp)
}

fn configure(w: &mut SpectrumWindow, cache: &Path, bars: usize) {
    w.mode = SpectrumMode::PreProcess;
    w.style = VizStyle::Bars;
    w.analyzer.cache_dir_override = Some(cache.to_path_buf());
    w.analyzer.cache_budget_gb = 0.0;
    w.analyzer.pre_stereo_enabled = true;
    w.bar_count = bars;
    w.analyzer.bar_count = bars;
    w.bar_mapping = BarMappingMode::Superlet;
    w.analyzer.bar_mapping = BarMappingMode::Superlet;
    w.min_freq = 20.0;
    w.max_freq = 24_000.0;
    w.analyzer.min_freq = 20.0;
    w.analyzer.max_freq = 24_000.0;
    w.analyzer.aslt_preset = None;
    let mut c = aslt::AsltPreset::Standard.config();
    c.n_wavelets = 9;
    c.q_ratio = 2.0;
    c.scale = freq_scale::FreqScale::Blend(0.5);
    c.spread = 0.5;
    c.max_window_s = 0.25;
    w.analyzer.aslt_cfg = c;
    w.analyzer.pre_fps = 180.0;
    w.analyzer.dsd_rate = 88_200;
}

#[test]
#[ignore]
fn r13_lifecycle_soak() {
    let hours: f64 = std::env::var("MOOSIK_R13_SOAK_HOURS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05);
    let out = std::env::var("MOOSIK_R13_SOAK_OUT").expect("MOOSIK_R13_SOAK_OUT");
    let mut seed: u64 = std::env::var("MOOSIK_R13_SOAK_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(0x5EED_1234);
    let dir = std::env::temp_dir().join(format!("moosik_r13_soak_{}", std::process::id()));
    let cache = dir.join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    let _calib = gpu_calib::isolated_test_env("r13_soak");
    let corpus: Vec<(PathBuf, u32)> = vec![
        (dir.join("a44.wav"), 44_100), (dir.join("b48.wav"), 48_000), (dir.join("c96.wav"), 96_000), (dir.join("d64.dsf"), 88_200),
    ];
    write_wav(&corpus[0].0, 44_100, 12.0, 440.0, 1250.0, 1);
    write_wav(&corpus[1].0, 48_000, 10.0, 330.0, 2000.0, 2);
    write_wav(&corpus[2].0, 96_000, 8.0, 880.0, 5000.0, 3);
    super::dsd_rate_tests::write_dsf(&corpus[3].0, 2_822_400, 8.0, 700.0, 1700.0);
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(&out).unwrap();
    let mut emit = |s: String| { use std::io::Write; writeln!(log, "{s}").unwrap(); log.flush().unwrap(); };
    let mut rnd = move |m: u64| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed % m.max(1) };
    let deadline = Instant::now() + Duration::from_secs_f64(hours * 3600.0);
    let mut w = SpectrumWindow::new_isolated();
    let bars_gen = [1024usize, 512];
    let mut generation = 0usize;
    configure(&mut w, &cache, bars_gen[generation]);
    // (track, generation, channels resident) -> (frames, hash) of the first completion.
    type Installs = std::collections::HashMap<(usize, usize, bool), (usize, u64, u64, u64)>;
    let mut seen: Installs = Default::default();
    let mut clock: f64;
    let frame = |w: &mut SpectrumWindow, clock: &mut f64, playing: bool| {
        if playing { *clock += 1.0 / 60.0; }
        w.last_fft_time = Some(Instant::now() - Duration::from_millis(20));
        w.tick(*clock, playing);
        std::thread::sleep(Duration::from_millis(2));
    };
    let r0 = resources();
    emit(format!("{{\"ev\":\"start\",\"hours\":{hours},\"commit\":{},\"ws\":{},\"handles\":{},\"threads\":{}}}", r0.0, r0.1, r0.2, r0.3));
    let mut it = 0u64;
    // Separately counted work (closeout).
    let (mut n_starts, mut n_fresh, mut n_aborted, mut n_reuse) = (0usize, 0usize, 0usize, 0usize);
    let fail = |emit: &mut dyn FnMut(String), it: u64, what: String| -> ! {
        emit(format!("{{\"ev\":\"failure\",\"iter\":{it},\"what\":{}}}", serde_json::Value::String(what.clone())));
        panic!("soak failure at iteration {it}: {what}");
    };
    while Instant::now() < deadline {
        it += 1;
        let t = rnd(corpus.len() as u64) as usize;
        let action = rnd(100);
        let (path, sr) = (&corpus[t].0, corpus[t].1);
        let t_iter = Instant::now();
        let mut what = String::new();
        let mut cancel_latency_ms = -1.0f64;
        let mut completed = false;
        // Occasionally change the settings generation (new cache key).
        if action < 5 {
            generation = 1 - generation;
            w.on_stop();
            configure(&mut w, &cache, bars_gen[generation]);
            what.push_str("generation;");
        }
        let split = rnd(2) == 0;
        w.channel_view = if split { channels::ChannelView::Split } else { channels::ChannelView::Mix };
        clock = 0.0;
        let (o0, s0) = (outcome_count(&w), spawned(&w));
        w.on_play(path, sr);
        if action < 45 {
            // Run to completion, with seeks / pauses / view toggles on the way.
            what.push_str("complete;");
            let until = Instant::now() + Duration::from_secs(300);
            let mut quiet = 0;
            while quiet < 3 {
                let r = rnd(1000);
                if r < 3 { w.on_seek(rnd(8) as f64); what.push_str("seek;"); }
                let playing = r % 7 != 0;
                frame(&mut w, &mut clock, playing);
                quiet = if w.analyzer.analysis_idle() { quiet + 1 } else { 0 };
                if Instant::now() > until { fail(&mut emit, it, format!("analysis of {} never finished", path.display())); }
            }
            if split && let Err(e) = require_resident(&mut w, &mut clock, Duration::from_secs(300)) {
                fail(&mut emit, it, e);
            }
            completed = true;
        } else if action < 60 {
            // Cache reuse candidate: play, which loads from cache if present.
            what.push_str("replay;");
            let until = Instant::now() + Duration::from_secs(300);
            let mut quiet = 0;
            while quiet < 3 {
                frame(&mut w, &mut clock, true);
                quiet = if w.analyzer.analysis_idle() { quiet + 1 } else { 0 };
                if Instant::now() > until { fail(&mut emit, it, "replay never settled".into()); }
            }
            if split && let Err(e) = require_resident(&mut w, &mut clock, Duration::from_secs(300)) {
                fail(&mut emit, it, e);
            }
            completed = true;
        } else {
            // Cancel at a random point: by switching track, by stop, or by a view change then stop.
            let wait = Duration::from_millis(20 + rnd(4000));
            let until = Instant::now() + wait;
            while Instant::now() < until {
                frame(&mut w, &mut clock, true);
                if rnd(200) == 0 { w.channel_view = if w.channel_view == channels::ChannelView::Mix { channels::ChannelView::Split } else { channels::ChannelView::Mix }; what.push_str("view;"); }
            }
            if action < 80 {
                let other = (t + 1 + rnd(corpus.len() as u64 - 1) as usize) % corpus.len();
                what.push_str("switch;");
                w.on_play(&corpus[other].0, corpus[other].1);
                let until = Instant::now() + Duration::from_millis(rnd(1500));
                while Instant::now() < until { frame(&mut w, &mut clock, true); }
            } else {
                what.push_str("stop;");
            }
        }
        // A run to completion is either a fresh completion (one successful
        // producer result for this track) or a cache reuse (no worker
        // started); anything else, or nothing installed, fails.
        let fresh_here = spawned(&w) > s0;
        if completed {
            let new = outcomes_since(&w, o0);
            let checked = if fresh_here {
                verify_fresh(&w, path, &new, bars_gen[generation])
            } else if new.is_empty() {
                verify_installed(&w, None, bars_gen[generation])
            } else {
                Err(format!("worker results without a worker start: {new:?}"))
            };
            let checked = match checked {
                // Mix-only views need not have channels resident.
                Err(e) if e == "channels not resident" && !split => Ok(installed_codes_mix(&w)),
                other => other,
            };
            if let Err(e) = checked { fail(&mut emit, it, format!("completion intent: {e}")); }
            if fresh_here { n_fresh += 1 } else { n_reuse += 1 }
        }
        // Record a completed install against its first completion.
        if completed {
            let resident = w.analyzer.pre_channels_resident();
            let (n, h) = hash_frames(&w.analyzer.pre_frames);
            let (hl, hr) = if resident {
                let p = w.analyzer.pre_stereo().expect("resident").clone();
                (hash_frames(p.left()).1, hash_frames(p.right()).1)
            } else { (0, 0) };
            match seen.get(&(t, generation, resident)) {
                None => { seen.insert((t, generation, resident), (n, h, hl, hr)); }
                Some(&(n0, h0, hl0, hr0)) => {
                    if (n, h, hl, hr) != (n0, h0, hl0, hr0) {
                        fail(&mut emit, it, format!("install of track {t} generation {generation} resident {resident} differs from its first completion: {:?} vs {:?}", (n, h, hl, hr), (n0, h0, hl0, hr0)));
                    }
                }
            }
        }
        // Stop and require quiescence.
        let t_stop = Instant::now();
        w.on_stop();
        if let Err(e) = quiesce(&mut w, &mut clock, Duration::from_secs(60)) { fail(&mut emit, it, e); }
        cancel_latency_ms = cancel_latency_ms.max(t_stop.elapsed().as_secs_f64() * 1e3);
        n_starts += spawned(&w) - s0;
        n_aborted += outcomes_since(&w, o0).iter().filter(|o| o.kind == "aborted").count();
        let (bytes, tmp) = dir_bytes(&cache);
        if tmp > 0 { fail(&mut emit, it, format!("{tmp} temporary cache file(s) left after quiescence")); }
        let r = resources();
        emit(format!("{{\"ev\":\"iter\",\"iter\":{it},\"track\":{t},\"generation\":{generation},\"split\":{split},\"actions\":\"{what}\",\"completed\":{completed},\
                       \"iter_ms\":{:.0},\"quiet_after_stop_ms\":{cancel_latency_ms:.1},\"commit\":{},\"ws\":{},\"handles\":{},\"threads\":{},\
                       \"cache_bytes\":{bytes},\"distinct_installs\":{},\"fresh_here\":{fresh_here},\"starts\":{n_starts},\"fresh\":{n_fresh},\"aborted\":{n_aborted},\"reuse\":{n_reuse}}}",
                      t_iter.elapsed().as_secs_f64() * 1e3, r.0, r.1, r.2, r.3, seen.len()));
    }
    // Final quiescence sample after a pause.
    std::thread::sleep(Duration::from_secs(5));
    let r = resources();
    emit(format!("{{\"ev\":\"end\",\"iters\":{it},\"commit\":{},\"ws\":{},\"handles\":{},\"threads\":{},\"cache_bytes\":{}}}", r.0, r.1, r.2, r.3, dir_bytes(&cache).0));
    drop(w);
    let _ = std::fs::remove_dir_all(&dir);
}

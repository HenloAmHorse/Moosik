//! R13 closeout: a short, quota-gated mixed lifecycle campaign whose fresh
//! work is load-bearing, and the checks it shares with the long soak
//! (`r13_soak_tests`). Headless `SpectrumWindow` pre-process lifecycle only;
//! not GUI, audio or rendering evidence.
//!
//! R13's ten-hour soak counted an iteration "completed" whenever frames were
//! installed, so after its first few analyses it was a cache-replay soak (the
//! producer log shows 11 starts, 8 fresh completions, 3 aborts). Here every
//! kind of work is counted separately and each intent is asserted:
//! - a **fresh completion** needs a successful producer result (this worker's
//!   `WorkerOutcome`, "done") for the requested track, with frames, the
//!   requested bar width and channels; the installed result must be that
//!   track's at the selected settings (`pre_cache_path` equals the window's
//!   `current_cache_path`), have the producer's frame count, and the exact
//!   Mix/Left/Right codes of the first fresh completion of that track;
//! - a **cache reuse** must install without any worker starting, with the
//!   trusted codes, and leave the cache files byte-identical;
//! - an **active cancellation** is issued only after this job has been seen
//!   on the live-row column pass (C2) — the process-wide pass counter of the
//!   diagnostic build, read while this campaign is the only analysis in the
//!   process — and counts only if the producer then reports "aborted". The
//!   worker must retire, nothing may be published for it, nothing stale may
//!   be installed, and a complete cache beside it must stay byte-identical;
//! - a requested channel view must actually become resident: a timeout is a
//!   failure, never a fall-through;
//! - quiescence is the analysis slot free, no live analysis worker, and
//!   every sidecar reader this analyser started finished.
//!
//! Every failure is written with its reason before the test panics.

use super::*;
use std::time::{Duration, Instant};

/// FNV-1a over every row: (frames, hash).
pub(super) fn hash_frames(p: &cache::PreFrames) -> (usize, u64) {
    let mut h = 0xcbf29ce484222325u64;
    for i in 0..p.len() {
        for v in p.row(i).unwrap_or(&[]) {
            h = (h ^ *v as u64).wrapping_mul(0x100000001b3);
        }
    }
    (p.len(), h)
}

/// Installed codes: (frames, Mix, Left, Right); Left/Right 0 when not resident.
pub(super) fn installed_codes(w: &SpectrumWindow) -> (usize, u64, u64, u64) {
    let (n, h) = hash_frames(&w.analyzer.pre_frames);
    let (l, r) = match w.analyzer.pre_stereo() {
        Some(p) => (hash_frames(p.left()).1, hash_frames(p.right()).1),
        None => (0, 0),
    };
    (n, h, l, r)
}

/// One UI frame: tick and give the workers a moment.
pub(super) fn frame(w: &mut SpectrumWindow, clock: &mut f64, playing: bool) {
    if playing { *clock += 1.0 / 60.0; }
    w.last_fft_time = Some(Instant::now() - Duration::from_millis(20));
    w.tick(*clock, playing);
    std::thread::sleep(Duration::from_millis(2));
}

/// Worker results recorded since `from`.
pub(super) fn outcomes_since(w: &SpectrumWindow, from: usize) -> Vec<WorkerOutcome> {
    w.analyzer.lifecycle_probe.outcomes.lock().unwrap()[from..].to_vec()
}

pub(super) fn outcome_count(w: &SpectrumWindow) -> usize {
    w.analyzer.lifecycle_probe.outcomes.lock().unwrap().len()
}

pub(super) fn spawned(w: &SpectrumWindow) -> usize {
    w.analyzer.lifecycle_probe.spawned.load(std::sync::atomic::Ordering::SeqCst)
}

/// Sidecar readers started but not finished.
pub(super) fn readers_live(w: &SpectrumWindow) -> usize {
    let s = &w.analyzer.reader_stats;
    s.started.load(std::sync::atomic::Ordering::SeqCst) - s.finished.load(std::sync::atomic::Ordering::SeqCst)
}

/// Ticks (not playing) until the analysis slot is free, no analysis worker
/// is alive and every sidecar reader has finished. Err after `limit`.
/// Returns how long it took.
pub(super) fn quiesce(w: &mut SpectrumWindow, clock: &mut f64, limit: Duration) -> Result<Duration, String> {
    let t0 = Instant::now();
    loop {
        w.analyzer.service_preprocess();
        frame(w, clock, false);
        let live = w.analyzer.lifecycle_probe.live.load(std::sync::atomic::Ordering::SeqCst);
        let readers = readers_live(w);
        if !w.analyzer.analysis_active() && live == 0 && readers == 0 {
            return Ok(t0.elapsed());
        }
        if t0.elapsed() > limit {
            return Err(format!("not quiet {limit:?} after stop: slot active {}, live workers {live}, live readers {readers}",
                               w.analyzer.analysis_active()));
        }
    }
}

/// Ticks (playing) until the channels are resident. Err after `limit`:
/// a requested channel state that never arrives fails the gate.
pub(super) fn require_resident(w: &mut SpectrumWindow, clock: &mut f64, limit: Duration) -> Result<(), String> {
    let t0 = Instant::now();
    while !w.analyzer.pre_channels_resident() {
        if t0.elapsed() > limit {
            return Err(format!("requested channels not resident after {limit:?}"));
        }
        frame(w, clock, true);
    }
    Ok(())
}

/// Ticks (playing) until no analysis is running or waiting, three frames in a row.
#[cfg_attr(not(feature = "leaf-diagnostics"), allow(dead_code))] // used by the diagnostic-build campaign
pub(super) fn settle(w: &mut SpectrumWindow, clock: &mut f64, limit: Duration) -> Result<(), String> {
    let t0 = Instant::now();
    let mut quiet = 0;
    while quiet < 3 {
        frame(w, clock, true);
        quiet = if w.analyzer.analysis_idle() { quiet + 1 } else { 0 };
        if t0.elapsed() > limit {
            return Err(format!("analysis never settled within {limit:?}"));
        }
    }
    Ok(())
}

/// A fresh completion of `track` at `bars`: the new worker results must be
/// exactly one successful one for this track (frames, width, channels), and
/// what is installed must be it, at the selected settings. Returns the
/// installed codes.
pub(super) fn verify_fresh(w: &SpectrumWindow, track: &Path, new: &[WorkerOutcome], bars: usize)
    -> Result<(usize, u64, u64, u64), String>
{
    let [o] = new else {
        return Err(format!("expected exactly one worker result for a fresh completion, got {}: {new:?}", new.len()));
    };
    if o.kind != "done" {
        return Err(format!("fresh completion of {} ended '{}'", track.display(), o.kind));
    }
    if o.track != track {
        return Err(format!("worker result is for {}, requested {}", o.track.display(), track.display()));
    }
    if o.frames == 0 || o.width != bars || !o.stereo {
        return Err(format!("worker result shape: {} frames, width {} (want {bars}), channels {}", o.frames, o.width, o.stereo));
    }
    verify_installed(w, Some(o.frames), bars)
}

/// How a cancellation ended, from the producer's own result.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(super) enum Cancel {
    /// Issued after the C2 witness and the producer reported "aborted": the
    /// only kind that counts toward the active-cancellation quota.
    WitnessedAbort,
    /// The producer reported "aborted", but the cancel was not witnessed.
    UnwitnessedAbort,
    /// The producer finished, with a valid result, before the cancel reached it.
    CompletedRace,
}

/// Classifies the worker results recorded after a cancelled job retired.
/// Exactly one result, for `track`, is required: "aborted" is a cancellation
/// (witnessed only if `witnessed`), "done" with a valid shape a completed
/// race; an error, a missing, duplicate or wrong-track result, a malformed
/// "done" and an unknown kind all fail. Used by every cancellation branch of
/// the campaign.
pub(super) fn classify_cancel(track: &Path, new: &[WorkerOutcome], bars: usize, witnessed: bool) -> Result<Cancel, String> {
    let [o] = new else {
        return Err(format!("expected exactly one worker result for a cancelled job, got {}: {new:?}", new.len()));
    };
    if o.track != track {
        return Err(format!("cancelled worker result is for {}, cancelled {}", o.track.display(), track.display()));
    }
    match o.kind {
        "aborted" if witnessed => Ok(Cancel::WitnessedAbort),
        "aborted" => Ok(Cancel::UnwitnessedAbort),
        "done" if o.frames > 0 && o.width == bars && o.stereo => Ok(Cancel::CompletedRace),
        "done" => Err(format!("completed race with invalid shape: {} frames, width {} (want {bars}), channels {}", o.frames, o.width, o.stereo)),
        "error" => Err(format!("cancelled analysis of {} ended 'error'", track.display())),
        k => Err(format!("unknown worker outcome '{k}'")),
    }
}

/// What is installed belongs to the window's current track at the selected
/// settings, with `frames` frames (when given) of `bars` bars, and channels
/// resident. Returns the installed codes.
pub(super) fn verify_installed(w: &SpectrumWindow, frames: Option<usize>, bars: usize) -> Result<(usize, u64, u64, u64), String> {
    let p = &w.analyzer.pre_frames;
    if p.is_empty() {
        return Err("nothing installed".into());
    }
    if let Some(n) = frames && p.len() != n {
        return Err(format!("installed {} frames, producer sent {n}", p.len()));
    }
    if p.row(0).map_or(0, |r| r.len()) != bars {
        return Err(format!("installed width {:?}, want {bars}", p.row(0).map(|r| r.len())));
    }
    let want = w.current_cache_path();
    if w.analyzer.pre_cache_path().map(Path::to_path_buf) != want {
        return Err(format!("installed {:?} is not the selected track/settings' {want:?}", w.analyzer.pre_cache_path()));
    }
    if !w.analyzer.pre_channels_resident() {
        return Err("channels not resident".into());
    }
    Ok(installed_codes(w))
}

/// Every file in `dir` with its bytes.
#[cfg_attr(not(feature = "leaf-diagnostics"), allow(dead_code))] // used by the diagnostic-build campaign
pub(super) fn snapshot(dir: &Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(dir).map(|rd| rd.flatten()
        .filter(|e| e.path().is_file())
        .map(|e| (e.file_name().to_string_lossy().into_owned(), std::fs::read(e.path()).unwrap()))
        .collect()).unwrap_or_default()
}

/// What the campaign achieved.
#[derive(Default, Debug, Clone)]
#[cfg_attr(not(feature = "leaf-diagnostics"), allow(dead_code))] // used by the diagnostic-build campaign
pub(super) struct Counts {
    pub fresh_starts: usize,
    pub fresh_ok: usize,
    pub fresh_ok_dsd: usize,
    pub cancels_issued: usize,
    /// Issued after the C2 witness, and the producer reported "aborted".
    pub cancels_active: usize,
    pub cancels_active_dsd: usize,
    /// Issued but the producer had already finished (not counted as active).
    pub cancels_raced: usize,
    pub reuse_ok: usize,
}

/// The closeout quotas.
pub(super) fn check_quotas(c: &Counts) -> Result<(), String> {
    let mut short = Vec::new();
    if c.fresh_ok < 12 { short.push(format!("fresh completions {} < 12", c.fresh_ok)); }
    if c.fresh_ok_dsd < 1 { short.push("no DSD fresh completion".into()); }
    if c.cancels_active < 12 { short.push(format!("witnessed active cancellations {} < 12", c.cancels_active)); }
    if c.cancels_active_dsd < 1 { short.push("no DSD active cancellation".into()); }
    if c.reuse_ok < 12 { short.push(format!("verified cache reuses {} < 12", c.reuse_ok)); }
    if short.is_empty() { Ok(()) } else { Err(short.join("; ")) }
}

#[cfg_attr(not(feature = "leaf-diagnostics"), allow(dead_code))] // used by the diagnostic-build campaign
fn configure(w: &mut SpectrumWindow, cache: &Path) {
    w.mode = SpectrumMode::PreProcess;
    w.style = VizStyle::Bars;
    w.analyzer.cache_dir_override = Some(cache.to_path_buf());
    w.analyzer.cache_budget_gb = 0.0;
    w.analyzer.pre_stereo_enabled = true;
    w.bar_count = BARS;
    w.analyzer.bar_count = BARS;
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
    w.channel_view = channels::ChannelView::Split;
}

const BARS: usize = 1024;

/// The campaign. Ignored; run alone (`--exact`, one test thread) in a
/// diagnostic build under the closeout runner: `MOOSIK_R13X_OUT` (JSONL),
/// `MOOSIK_R13X_DIR` (private working directory, created empty),
/// `MOOSIK_R13X_BUDGET_S` (default 1500).
#[cfg(feature = "leaf-diagnostics")]
#[test]
#[ignore]
fn r13x_fresh_work_gate() {
    use rustfft::moosik_diagnostics as d;
    let out = std::env::var("MOOSIK_R13X_OUT").expect("MOOSIK_R13X_OUT");
    let dir = PathBuf::from(std::env::var("MOOSIK_R13X_DIR").expect("MOOSIK_R13X_DIR"));
    let budget = Duration::from_secs_f64(std::env::var("MOOSIK_R13X_BUDGET_S").ok().and_then(|v| v.parse().ok()).unwrap_or(1500.0));
    assert!(!dir.exists() || std::fs::read_dir(&dir).unwrap().next().is_none(), "working directory not empty");
    std::fs::create_dir_all(&dir).unwrap();
    let _calib = gpu_calib::isolated_test_env("r13x_fresh");
    gpu_calib::set_workers(6);
    let t_start = Instant::now();
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(&out).unwrap();
    let mut emit = |s: String| { use std::io::Write; writeln!(log, "{s}").unwrap(); log.flush().unwrap(); };
    let fail = |emit: &mut dyn FnMut(String), phase: &str, i: usize, what: String| -> ! {
        emit(format!("{{\"ev\":\"failure\",\"phase\":\"{phase}\",\"i\":{i},\"what\":{}}}", serde_json::Value::String(what.clone())));
        panic!("{phase} {i}: {what}");
    };
    let res = |w: &SpectrumWindow| {
        let r = super::r13_soak_tests::resources();
        format!("\"commit\":{},\"ws\":{},\"handles\":{},\"threads\":{},\"spawned\":{}", r.0, r.1, r.2, r.3, spawned(w))
    };
    // Short fixtures; the DSD one decodes to the 88.2 kHz analysis rate.
    let corpus: Vec<(PathBuf, u32, &str, f64)> = vec![
        (dir.join("d64.dsf"), 88_200, "dsd", 8.0),
        (dir.join("a44.wav"), 44_100, "pcm44", 8.0),
        (dir.join("c96.wav"), 96_000, "pcm96", 6.0),
    ];
    super::dsd_rate_tests::write_dsf(&corpus[0].0, 2_822_400, corpus[0].3, 700.0, 1700.0);
    super::r13_soak_tests::write_wav(&corpus[1].0, 44_100, corpus[1].3, 440.0, 1250.0, 1);
    super::r13_soak_tests::write_wav(&corpus[2].0, 96_000, corpus[2].3, 880.0, 5000.0, 3);
    emit(format!("{{\"ev\":\"start\",\"bars\":{BARS},\"n_wavelets\":9,\"q_ratio\":2.0,\"fps\":180,\"dsd_rate\":88200,\"workers_requested\":6,\"workers_selected\":{},\"cores\":{},\"budget_s\":{}}}",
                 gpu_calib::workers(), gpu_calib::core_count(), budget.as_secs_f64()));
    let mut w = SpectrumWindow::new_isolated();
    let mut clock: f64;
    let mut c = Counts::default();
    let mut slot = 0usize;
    let mut next_dir = |tag: &str| { slot += 1; let p = dir.join(format!("cache_{slot:03}_{tag}")); std::fs::create_dir_all(&p).unwrap(); p };
    // track index -> (trusted codes, a directory holding its complete cache)
    type Trusted = Option<((usize, u64, u64, u64), PathBuf)>;
    let mut trusted: Vec<Trusted> = vec![None; corpus.len()];
    let limit = Duration::from_secs(300);

    // Fresh: DSD x6, 44.1 kHz x4, 96 kHz x4, each into its own new cache directory.
    let fresh_plan = [0usize, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0];
    for (i, &t) in fresh_plan.iter().enumerate() {
        let (path, sr, kind, secs) = &corpus[t];
        let cache = next_dir(&format!("fresh_{kind}"));
        configure(&mut w, &cache);
        let (o0, s0) = (outcome_count(&w), spawned(&w));
        let t0 = Instant::now();
        clock = 0.0;
        w.on_play(path, *sr);
        if let Err(e) = settle(&mut w, &mut clock, limit) { fail(&mut emit, "fresh", i, e); }
        if spawned(&w) != s0 + 1 { fail(&mut emit, "fresh", i, format!("{} workers started, want 1", spawned(&w) - s0)); }
        c.fresh_starts += 1;
        let codes = match verify_fresh(&w, path, &outcomes_since(&w, o0), BARS) {
            Ok(v) => v,
            Err(e) => fail(&mut emit, "fresh", i, e),
        };
        let nominal = secs * 180.0;
        if (codes.0 as f64 - nominal).abs() > nominal * 0.01 + 2.0 {
            fail(&mut emit, "fresh", i, format!("{} frames for {secs} s at 180 fps", codes.0));
        }
        match &trusted[t] {
            None => trusted[t] = Some((codes, cache.clone())),
            Some((want, _)) if *want != codes => fail(&mut emit, "fresh", i, format!("codes {codes:?} differ from the first fresh completion {want:?}")),
            Some(_) => {}
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        w.on_stop();
        let q = quiesce(&mut w, &mut clock, Duration::from_secs(60)).unwrap_or_else(|e| fail(&mut emit, "fresh", i, e));
        if snapshot(&cache).keys().any(|k| k.contains(".tmp-")) { fail(&mut emit, "fresh", i, "temporary file after quiescence".into()); }
        c.fresh_ok += 1;
        if *kind == "dsd" { c.fresh_ok_dsd += 1; }
        emit(format!("{{\"ev\":\"fresh\",\"i\":{i},\"track\":\"{kind}\",\"frames\":{},\"mix\":\"{:016x}\",\"left\":\"{:016x}\",\"right\":\"{:016x}\",\"ms\":{ms:.0},\"quiet_ms\":{:.1},{}}}",
                     codes.0, codes.1, codes.2, codes.3, q.as_secs_f64() * 1e3, res(&w)));
    }

    // Reuse: a private copy of a trusted complete cache, in a new directory each time.
    for i in 0..12 {
        let t = i % corpus.len();
        let (path, sr, kind, _) = &corpus[t];
        let (want, seed) = trusted[t].clone().expect("trusted");
        let cache = next_dir(&format!("reuse_{kind}"));
        for (name, bytes) in snapshot(&seed) { std::fs::write(cache.join(name), bytes).unwrap(); }
        let before = snapshot(&cache);
        configure(&mut w, &cache);
        let (o0, s0) = (outcome_count(&w), spawned(&w));
        clock = 0.0;
        w.on_play(path, *sr);
        let mut busy = false;
        let t0 = Instant::now();
        while !(w.analyzer.analysis_idle() && w.analyzer.pre_channels_resident()) {
            frame(&mut w, &mut clock, true);
            busy |= w.analyzer.analysis_busy();
            if t0.elapsed() > Duration::from_secs(60) { fail(&mut emit, "reuse", i, "cache reuse never installed resident channels".into()); }
        }
        if busy || spawned(&w) != s0 || outcome_count(&w) != o0 {
            fail(&mut emit, "reuse", i, format!("analysis ran during an intended reuse (busy {busy}, workers {})", spawned(&w) - s0));
        }
        let codes = verify_installed(&w, Some(want.0), BARS).unwrap_or_else(|e| fail(&mut emit, "reuse", i, e));
        if codes != want { fail(&mut emit, "reuse", i, format!("reused codes {codes:?} differ from trusted {want:?}")); }
        w.on_stop();
        let q = quiesce(&mut w, &mut clock, Duration::from_secs(60)).unwrap_or_else(|e| fail(&mut emit, "reuse", i, e));
        if snapshot(&cache) != before { fail(&mut emit, "reuse", i, "cache files changed by a reuse".into()); }
        c.reuse_ok += 1;
        emit(format!("{{\"ev\":\"reuse\",\"i\":{i},\"track\":\"{kind}\",\"frames\":{},\"ms\":{:.0},\"quiet_ms\":{:.1},{}}}",
                     codes.0, t0.elapsed().as_secs_f64() * 1e3, q.as_secs_f64() * 1e3, res(&w)));
    }

    // Active cancellation of DSD analyses, after the C2 witness: alternately by
    // stop, and by switching to the 44.1 kHz track whose complete cache sits in
    // the same directory (it must install from that cache, untouched).
    let mut i = 0usize;
    while c.cancels_active < 14 && t_start.elapsed() < budget {
        i += 1;
        let by_switch = i.is_multiple_of(2);
        let (path, sr, _, _) = &corpus[0];
        let cache = next_dir(if by_switch { "cancel_switch" } else { "cancel_stop" });
        let (want_b, seed_b) = trusted[1].clone().expect("trusted");
        if by_switch { for (name, bytes) in snapshot(&seed_b) { std::fs::write(cache.join(name), bytes).unwrap(); } }
        let before = snapshot(&cache);
        configure(&mut w, &cache);
        let (o0, s0) = (outcome_count(&w), spawned(&w));
        let _ = d::passes_take();
        clock = 0.0;
        w.on_play(path, *sr);
        // Witness: this job on the live-row column pass, and still running.
        let t0 = Instant::now();
        let mut live = 0u64;
        while live == 0 {
            frame(&mut w, &mut clock, true);
            live += d::passes_take().0;
            if outcome_count(&w) > o0 { break; }
            if t0.elapsed() > Duration::from_secs(60) { fail(&mut emit, "cancel", i, "no live-row column pass within 60 s".into()); }
        }
        let witnessed_ms = t0.elapsed().as_secs_f64() * 1e3;
        let running = outcome_count(&w) == o0 && w.analyzer.analysis_busy();
        if live == 0 || !running {
            // Not witnessed while running: never counted, but its result must
            // still be a valid abort or a valid completed race.
            w.on_stop();
            quiesce(&mut w, &mut clock, Duration::from_secs(60)).unwrap_or_else(|e| fail(&mut emit, "cancel", i, e));
            let new = outcomes_since(&w, o0);
            if spawned(&w) != s0 + 1 { fail(&mut emit, "cancel", i, format!("{} workers started, want 1", spawned(&w) - s0)); }
            let class = classify_cancel(path, &new, BARS, false).unwrap_or_else(|e| fail(&mut emit, "cancel", i, e));
            if !w.analyzer.pre_frames.is_empty() || w.analyzer.pre_cache_path().is_some() {
                fail(&mut emit, "cancel", i, "something is installed after stop".into());
            }
            let after = snapshot(&cache);
            if after.keys().any(|k| k.contains(".tmp-")) { fail(&mut emit, "cancel", i, "temporary file after quiescence".into()); }
            for (k, v) in &before { if after.get(k) != Some(v) { fail(&mut emit, "cancel", i, format!("{k} changed")); } }
            if class == Cancel::UnwitnessedAbort && after != before { fail(&mut emit, "cancel", i, "an aborted analysis changed the directory".into()); }
            emit(format!("{{\"ev\":\"cancel_skipped\",\"i\":{i},\"live_passes\":{live},\"running\":{running},\"outcome\":\"{class:?}\"}}"));
            continue;
        }
        c.cancels_issued += 1;
        let t_cancel = Instant::now();
        if by_switch {
            w.on_play(&corpus[1].0, corpus[1].1);
            if let Err(e) = settle(&mut w, &mut clock, limit) { fail(&mut emit, "cancel", i, e); }
            if let Err(e) = require_resident(&mut w, &mut clock, Duration::from_secs(60)) { fail(&mut emit, "cancel", i, e); }
            let codes = verify_installed(&w, Some(want_b.0), BARS).unwrap_or_else(|e| fail(&mut emit, "cancel", i, e));
            if codes != want_b { fail(&mut emit, "cancel", i, format!("switch target codes {codes:?} differ from trusted {want_b:?}")); }
        }
        w.on_stop();
        let q = quiesce(&mut w, &mut clock, Duration::from_secs(60)).unwrap_or_else(|e| fail(&mut emit, "cancel", i, e));
        let cancel_ms = t_cancel.elapsed().as_secs_f64() * 1e3;
        let new = outcomes_since(&w, o0);
        if spawned(&w) != s0 + 1 {
            fail(&mut emit, "cancel", i, format!("expected one worker, for the cancelled DSD track; got {} started, results {new:?}", spawned(&w) - s0));
        }
        let kind = classify_cancel(path, &new, BARS, true).unwrap_or_else(|e| fail(&mut emit, "cancel", i, e));
        if !w.analyzer.pre_frames.is_empty() || w.analyzer.pre_cache_path().is_some() {
            fail(&mut emit, "cancel", i, "something is installed after stop".into());
        }
        let after = snapshot(&cache);
        if after.keys().any(|k| k.contains(".tmp-")) { fail(&mut emit, "cancel", i, "temporary file after quiescence".into()); }
        match kind {
            Cancel::WitnessedAbort => {
                if after != before { fail(&mut emit, "cancel", i, format!("an aborted analysis changed the directory: {:?} -> {:?}", before.keys().collect::<Vec<_>>(), after.keys().collect::<Vec<_>>())); }
                c.cancels_active += 1;
                c.cancels_active_dsd += 1;
            }
            Cancel::CompletedRace => {
                // Finished before the cancel reached it: allowed to publish its
                // complete cache, never to leave the neighbour changed.
                c.cancels_raced += 1;
                for (k, v) in &before { if after.get(k) != Some(v) { fail(&mut emit, "cancel", i, format!("{k} changed")); } }
            }
            Cancel::UnwitnessedAbort => fail(&mut emit, "cancel", i, "witnessed cancel classified unwitnessed".into()),
        }
        emit(format!("{{\"ev\":\"cancel\",\"i\":{i},\"by\":\"{}\",\"outcome\":\"{}\",\"witness_ms\":{witnessed_ms:.0},\"live_passes\":{live},\"cancel_to_quiet_ms\":{cancel_ms:.1},\"quiet_ms\":{:.1},{}}}",
                     if by_switch { "switch" } else { "stop" }, new[0].kind, q.as_secs_f64() * 1e3, res(&w)));
    }

    // Two PCM (96 kHz) cancellations mid-analysis, for coverage beside DSD (not
    // counted in the witnessed quota: their top levels are dense).
    for j in 0..2 {
        let (path, sr, _, _) = &corpus[2];
        let cache = next_dir("cancel_pcm96");
        configure(&mut w, &cache);
        let (o0, s0) = (outcome_count(&w), spawned(&w));
        clock = 0.0;
        w.on_play(path, *sr);
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_millis(400) { frame(&mut w, &mut clock, true); }
        w.on_stop();
        let q = quiesce(&mut w, &mut clock, Duration::from_secs(60)).unwrap_or_else(|e| fail(&mut emit, "cancel_pcm", j, e));
        let new = outcomes_since(&w, o0);
        if spawned(&w) != s0 + 1 { fail(&mut emit, "cancel_pcm", j, format!("{} workers started, want 1", spawned(&w) - s0)); }
        // Not witnessed: an abort here never counts toward the active quota.
        let k = classify_cancel(path, &new, BARS, false).unwrap_or_else(|e| fail(&mut emit, "cancel_pcm", j, e));
        if !w.analyzer.pre_frames.is_empty() || w.analyzer.pre_cache_path().is_some() { fail(&mut emit, "cancel_pcm", j, "installed after stop".into()); }
        let after = snapshot(&cache);
        if after.keys().any(|k| k.contains(".tmp-")) { fail(&mut emit, "cancel_pcm", j, "temporary file after quiescence".into()); }
        if k == Cancel::UnwitnessedAbort && !after.is_empty() {
            fail(&mut emit, "cancel_pcm", j, "an aborted analysis published files".into());
        }
        emit(format!("{{\"ev\":\"cancel_pcm\",\"i\":{j},\"outcome\":\"{}\",\"class\":\"{k:?}\",\"quiet_ms\":{:.1}}}", new[0].kind, q.as_secs_f64() * 1e3));
    }

    let quotas = check_quotas(&c);
    std::thread::sleep(Duration::from_secs(2));
    let r = super::r13_soak_tests::resources();
    emit(format!("{{\"ev\":\"summary\",\"ok\":{},\"shortfall\":{},\"fresh_starts\":{},\"fresh_ok\":{},\"fresh_ok_dsd\":{},\"cancels_issued\":{},\"cancels_active\":{},\"cancels_active_dsd\":{},\"cancels_raced\":{},\"reuse_ok\":{},\"wall_s\":{:.1},\"end_commit\":{},\"end_handles\":{},\"end_threads\":{}}}",
                 quotas.is_ok(), serde_json::Value::String(quotas.clone().err().unwrap_or_default()),
                 c.fresh_starts, c.fresh_ok, c.fresh_ok_dsd, c.cancels_issued, c.cancels_active, c.cancels_active_dsd, c.cancels_raced, c.reuse_ok,
                 t_start.elapsed().as_secs_f64(), r.0, r.2, r.3));
    drop(w);
    if let Err(e) = quotas { panic!("quota shortfall: {e}"); }
}

// Negative controls: the gate's checks reject what they must.

#[test]
fn a_cache_only_schedule_fails_the_fresh_work_quota() {
    let replay_only = Counts { reuse_ok: 36_000, ..Default::default() };
    let e = check_quotas(&replay_only).unwrap_err();
    assert!(e.contains("fresh completions 0 < 12") && e.contains("active cancellations 0 < 12"), "{e}");
    let enough = Counts { fresh_starts: 12, fresh_ok: 12, fresh_ok_dsd: 4, cancels_issued: 12, cancels_active: 12, cancels_active_dsd: 12, cancels_raced: 0, reuse_ok: 12 };
    assert!(check_quotas(&enough).is_ok());
    assert!(check_quotas(&Counts { cancels_active_dsd: 0, ..enough.clone() }).is_err(), "PCM-only cancellations must not pass");
    assert!(check_quotas(&Counts { fresh_ok_dsd: 0, ..enough }).is_err(), "PCM-only completions must not pass");
}

#[test]
fn a_completion_error_never_counts_as_a_successful_install() {
    let _calib = gpu_calib::isolated_test_env("r13x_negative_error");
    let w = SpectrumWindow::new_isolated();
    let track = PathBuf::from("C:/nowhere/t.wav");
    let o = |kind| WorkerOutcome { track: track.clone(), kind, frames: 10, width: BARS, stereo: true };
    // The producer reported an error.
    let e = verify_fresh(&w, &track, &[o("error")], BARS).unwrap_err();
    assert!(e.contains("ended 'error'"), "{e}");
    let e = verify_fresh(&w, &track, &[o("aborted")], BARS).unwrap_err();
    assert!(e.contains("ended 'aborted'"), "{e}");
    // No producer result at all (settled without one).
    assert!(verify_fresh(&w, &track, &[], BARS).unwrap_err().contains("exactly one"));
    // A successful result for a different track.
    let other = WorkerOutcome { track: PathBuf::from("C:/nowhere/u.wav"), ..o("done") };
    assert!(verify_fresh(&w, &track, &[other], BARS).unwrap_err().contains("is for"));
    // A successful producer result with nothing installed.
    assert_eq!(verify_fresh(&w, &track, &[o("done")], BARS).unwrap_err(), "nothing installed");
}

#[test]
fn a_requested_channel_view_that_never_arrives_fails() {
    let _calib = gpu_calib::isolated_test_env("r13x_negative_resident");
    let mut w = SpectrumWindow::new_isolated();
    w.channel_view = channels::ChannelView::Split;
    let mut clock = 0.0;
    let e = require_resident(&mut w, &mut clock, Duration::from_millis(200)).unwrap_err();
    assert!(e.contains("not resident"), "{e}");
}

#[test]
fn cancellation_outcomes_fail_closed() {
    let track = PathBuf::from("C:/nowhere/d.dsf");
    let o = |kind| WorkerOutcome { track: track.clone(), kind, frames: 10, width: BARS, stereo: true };
    let aborted = WorkerOutcome { frames: 0, width: 0, stereo: false, ..o("aborted") };
    // Valid abort: witnessed counts, unwitnessed never becomes witnessed.
    assert_eq!(classify_cancel(&track, std::slice::from_ref(&aborted), BARS, true), Ok(Cancel::WitnessedAbort));
    assert_eq!(classify_cancel(&track, std::slice::from_ref(&aborted), BARS, false), Ok(Cancel::UnwitnessedAbort));
    // Valid completed race, witnessed or not; never an active cancellation.
    assert_eq!(classify_cancel(&track, &[o("done")], BARS, true), Ok(Cancel::CompletedRace));
    assert_eq!(classify_cancel(&track, &[o("done")], BARS, false), Ok(Cancel::CompletedRace));
    // A malformed completion is not a valid race.
    let e = classify_cancel(&track, &[WorkerOutcome { width: 7, ..o("done") }], BARS, true).unwrap_err();
    assert!(e.contains("invalid shape"), "{e}");
    // Error: the 4c8d770 rule (anything not "aborted" is a harmless race)
    // accepted it; the classifier rejects it.
    let error = WorkerOutcome { frames: 0, width: 0, stereo: false, ..o("error") };
    assert!(error.kind != "aborted", "the old rule would have counted this as cancels_raced");
    let e = classify_cancel(&track, std::slice::from_ref(&error), BARS, true).unwrap_err();
    assert!(e.contains("ended 'error'"), "{e}");
    // Missing and duplicate results.
    assert!(classify_cancel(&track, &[], BARS, true).unwrap_err().contains("got 0"));
    let e = classify_cancel(&track, &[aborted.clone(), aborted.clone()], BARS, true).unwrap_err();
    assert!(e.contains("got 2"), "{e}");
    // Wrong track, even with a valid abort.
    let other = WorkerOutcome { track: PathBuf::from("C:/nowhere/u.wav"), ..aborted.clone() };
    assert!(classify_cancel(&track, &[other], BARS, true).unwrap_err().contains("is for"));
    // Unknown outcome.
    let e = classify_cancel(&track, &[o("finished")], BARS, true).unwrap_err();
    assert!(e.contains("unknown worker outcome 'finished'"), "{e}");
}

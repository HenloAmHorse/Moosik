//! The pre-process lifecycle, under forced schedules.
//!
//! Every hand-off point is **established, not raced**. A worker that must still
//! be alive is parked by `analysis_hold` before it does any work. A worker that
//! must have finished and sent its result, without the UI having taken it yet,
//! is waited for through its own flag while the window is **not ticked**. Two
//! gates park a worker later: `publish_gate` after its last cancellation check
//! and before its cache is renamed into place, and `exit_gate` after it has
//! sent its result and before its thread exits. The fixtures are small, but no
//! test depends on one of them being fast or slow.
//!
//! Isolation: every window is `SpectrumWindow::new_isolated` (an empty preset
//! library, so nothing under the home directory is read or written), every
//! analyser has a private `cache_dir_override` in a temp folder, and the
//! inputs are generated WAV files there.
//!
//! Ported onto the exact-identity candidate (batch 1). A job's identity is
//! R15's `AnalysisSettings` at a sample rate, not a parallel field list.
//! **Policy difference from the lifecycle branch, deliberate:** editing a
//! setting never starts an analysis. A miss raises the manual "Re-analyze now"
//! banner, and a job running for the previous settings is left to finish (its
//! cache is valid for them; R14b). Only a track choice, Re-analyze, "Return to
//! my settings" and an explicit channel request ask for an analysis.
//!
//! The invariants, stated once:
//! - a track's newest request eventually runs without a manual Re-analyze;
//! - no picture other than the newest request's is ever installed;
//! - obsolete requests are coalesced, and at most one worker is alive;
//! - a picture that arrives while paused is drawn at the paused playhead;
//! - a repeat of the same track at the same settings keeps its work;
//! - failure and cancellation give the worker slot back;
//! - the slot is given back when the worker's thread has exited, not when its
//!   result arrives;
//! - no cache control acts while a worker, cancelled or not, can still publish;
//! - settings chosen in Real-time are offered for Re-analyze, not analysed,
//!   when Pre-process is selected;
//! - a cache control reports a deletion it could not make, and Re-analyze
//!   starts nothing unless the whole cache unit (Mix and sidecar) is gone.

use super::*;
use std::time::{Duration, Instant};

const SR: u32 = 44_100;
const BARS: usize = 32;
/// The paused playhead, in seconds.
const AT: f64 = 0.5;

struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "moosik_lifecycle_{tag}_{}_{:x}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0),
        ));
        std::fs::create_dir_all(&root).unwrap();
        Dir(root)
    }
}

impl Drop for Dir {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

fn write_wav(path: &Path, freqs: &[f32], seed: u32) {
    write_wav_noise(path, freqs, seed, 0.05)
}

fn write_wav_noise(path: &Path, freqs: &[f32], seed: u32, noise_amp: f32) {
    write_wav_secs(path, freqs, seed, noise_amp, 1);
}

fn write_wav_secs(path: &Path, freqs: &[f32], seed: u32, noise_amp: f32, secs: usize) {
    let n = SR as usize * secs;
    let mut x = seed.max(1);
    let mut data = Vec::with_capacity(n * 4);
    for i in 0..n {
        let t = i as f32 / SR as f32;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        let noise = (x as f32 / u32::MAX as f32) * 2.0 - 1.0;
        let tone: f32 = freqs.iter().map(|f| (std::f32::consts::TAU * f * t).sin()).sum();
        let l = tone * 0.25 / freqs.len() as f32 + noise * noise_amp;
        let r = tone * 0.2 / freqs.len() as f32 - noise * noise_amp;
        for s in [l, r] { data.extend(((s.clamp(-1.0, 1.0) * 32_767.0) as i16).to_le_bytes()) }
    }
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend(b"RIFF");
    out.extend(((36 + data.len()) as u32).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(2u16.to_le_bytes());
    out.extend(SR.to_le_bytes());
    out.extend((SR * 4).to_le_bytes());
    out.extend(4u16.to_le_bytes());
    out.extend(16u16.to_le_bytes());
    out.extend(b"data");
    out.extend((data.len() as u32).to_le_bytes());
    out.extend(data);
    std::fs::write(path, out).unwrap();
}

fn tracks(d: &Dir, n: usize) -> Vec<PathBuf> {
    (0..n).map(|i| {
        let p = d.0.join(format!("t{i}.wav"));
        write_wav(&p, &[300.0 + 170.0 * i as f32, 2100.0 + 390.0 * i as f32], 7 + i as u32);
        p
    }).collect()
}

fn cfg(q: f32) -> aslt::AsltConfig {
    let mut c = aslt::AsltPreset::Standard.config();
    c.q_ratio = q;
    c
}

/// A window in pre-process mode, Superlet, on a private cache directory.
fn window(cache: &Path, q: f32) -> SpectrumWindow {
    let mut w = SpectrumWindow::new_isolated();
    w.mode = SpectrumMode::PreProcess;
    w.analyzer.cache_dir_override = Some(cache.to_path_buf());
    w.analyzer.cache_budget_gb = 0.0;
    // Mix only unless a test asks for channels, so a sidecar is evidence.
    w.analyzer.pre_stereo_enabled = false;
    w.bar_count = BARS;
    w.analyzer.bar_count = BARS;
    w.bar_mapping = BarMappingMode::Superlet;
    w.analyzer.bar_mapping = BarMappingMode::Superlet;
    w.min_freq = 200.0;
    w.max_freq = 16_000.0;
    w.analyzer.min_freq = 200.0;
    w.analyzer.max_freq = 16_000.0;
    w.analyzer.aslt_preset = None;
    w.analyzer.aslt_cfg = cfg(q);
    // Mutation checks: one deliberate defect for the whole process.
    w.analyzer.lifecycle_probe.mutant = std::env::var("MOOSIK_LIFECYCLE_MUTANT").ok()
        .map(|m| &*Box::leak(m.into_boxed_str()));
    w
}

/// The Sharpness slider's own path: change the setting, then ask the window
/// whether a cache applies.
fn set_q(w: &mut SpectrumWindow, q: f32) {
    w.analyzer.aslt_cfg = cfg(q);
    w.try_load_or_flag_reanalysis();
}

/// One UI frame.
fn frame(w: &mut SpectrumWindow, playing: bool) {
    // `tick` throttles its playing half on wall-clock time; say a frame passed.
    w.last_fft_time = Some(Instant::now() - Duration::from_millis(20));
    w.tick(AT, playing);
}

/// Frames until nothing is running or waiting, watching every frame for a
/// picture that is neither nothing nor `newest`.
fn run(w: &mut SpectrumWindow, playing: bool, newest: &cache::PreFrames) -> (bool, usize) {
    let deadline = Instant::now() + Duration::from_secs(120);
    let (mut stale, mut quiet) = (0, 0);
    loop {
        frame(w, playing);
        let shown = &*w.analyzer.pre_frames;
        if !shown.is_empty() && shown != newest { stale += 1 }
        quiet = if w.analyzer.analysis_idle() { quiet + 1 } else { 0 };
        if quiet >= 3 { return (true, stale) }
        if Instant::now() > deadline { return (false, stale) }
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// A fresh analysis of `path` at `q`, in its own directory.
fn oracle(d: &Dir, path: &Path, q: f32) -> cache::PreFrames {
    let dir = d.0.join(format!("oracle_{}_{}", path.file_stem().unwrap().to_string_lossy(), q));
    let mut w = window(&dir, q);
    w.on_play(path, SR);
    let (ok, _) = run(&mut w, false, &cache::PreFrames::default());
    assert!(ok, "the oracle never finished");
    let f = (*w.analyzer.pre_frames).clone();
    assert!(!f.is_empty(), "the oracle produced nothing");
    f
}

/// What the bars must show at the paused playhead for `frames`, through
/// production's own snap.
fn row_at_playhead(frames: &cache::PreFrames, w: &SpectrumWindow) -> Vec<u32> {
    let mut r = window(&w.analyzer.cache_dir(), 1.0);
    r.analyzer.aslt_cfg = w.analyzer.aslt_cfg.clone();
    r.analyzer.sample_rate = SR;
    let rate = SR as f64 / r.analyzer.pre_hop() as f64;
    r.analyzer.set_pre_frames_with(frames.clone(), PreChannels::Missing, None, rate);
    r.analyzer.snap_pre_to(AT);
    r.analyzer.magnitudes.iter().map(|m| m.to_bits()).collect()
}

fn drawn(w: &SpectrumWindow) -> Vec<u32> { w.analyzer.magnitudes.iter().map(|m| m.to_bits()).collect() }

/// Park the next worker this window starts, before it does any work.
fn park(w: &mut SpectrumWindow) -> Arc<AtomicBool> {
    let hold = Arc::new(AtomicBool::new(true));
    w.analyzer.analysis_hold = Some(Arc::clone(&hold));
    hold
}

fn release(w: &mut SpectrumWindow, hold: &AtomicBool) {
    hold.store(false, Ordering::SeqCst);
    w.analyzer.analysis_hold = None;
}

/// Wait until the running worker has finished and sent its result, **without
/// ticking the window**, so the result is sitting unconsumed.
fn wait_exited_unconsumed(w: &SpectrumWindow) {
    let deadline = Instant::now() + Duration::from_secs(120);
    while w.analyzer.is_analyzing.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "the worker never finished");
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn spawned(w: &SpectrumWindow) -> usize { w.analyzer.lifecycle_probe.spawned.load(Ordering::SeqCst) }
fn max_live(w: &SpectrumWindow) -> usize { w.analyzer.lifecycle_probe.max_live.load(Ordering::SeqCst) }

// ── A finished result the UI has not taken yet ────────────────────────────

/// Settings change to one that has a cache while the previous request's
/// result is finished and waiting. The waiting result must never replace the
/// picture the new settings selected.
#[test]
fn an_unconsumed_result_never_replaces_a_cache_hit() {
    for playing in [false, true] {
        let d = Dir::new("hit");
        let t = tracks(&d, 1);
        let dir = d.0.join("cache");
        let mut w = window(&dir, 1.5);
        // A cache for q 1.5 first, so the change back to it is a hit.
        w.on_play(&t[0], SR);
        let (ok, _) = run(&mut w, playing, &cache::PreFrames::default());
        assert!(ok);
        let at_15 = (*w.analyzer.pre_frames).clone();
        // q 1.0 has no cache: its analysis runs, finishes, and waits.
        set_q(&mut w, 1.0);
        if w.analyzer.analysis_idle() {
            // A build that does not start the new request by itself: start it
            // the way the Re-analyze button does, so the schedule is the same.
            w.analyzer.start_preprocess(t[0].clone());
        }
        wait_exited_unconsumed(&w);
        // Back to q 1.5: a hit. The q 1.0 result is still waiting.
        set_q(&mut w, 1.5);
        assert!(*w.analyzer.pre_frames == at_15, "playing={playing}: the cache hit was not installed");
        let (ok, stale) = run(&mut w, playing, &at_15);
        assert!(ok, "playing={playing}: never settled");
        assert_eq!(stale, 0, "playing={playing}: the waiting result replaced the selected picture");
        assert!(*w.analyzer.pre_frames == at_15, "playing={playing}: the selected picture is gone");
    }
}

/// The same, then Stop. The old picture must never come back.
#[test]
fn an_unconsumed_result_never_returns_after_stop() {
    for playing in [false, true] {
        let d = Dir::new("stop");
        let t = tracks(&d, 1);
        let mut w = window(&d.0.join("cache"), 1.0);
        w.on_play(&t[0], SR);
        wait_exited_unconsumed(&w);
        w.on_stop();
        let (ok, stale) = run(&mut w, playing, &cache::PreFrames::default());
        assert!(ok, "playing={playing}: never settled");
        assert_eq!(stale, 0, "playing={playing}: a picture reappeared after Stop");
        assert!(w.analyzer.pre_frames.is_empty(), "playing={playing}: a picture is shown after Stop");
    }
}

// ── Requests while a worker is alive ──────────────────────────────────────

/// Four track changes while the first analysis is alive: the newest one runs
/// by itself, the two between are never started, no more than one worker is
/// alive at once, and nothing but the newest picture is ever installed. When
/// paused, the new picture is drawn at the paused playhead.
#[test]
fn track_changes_while_busy_run_only_the_newest() {
    for playing in [false, true] {
        let d = Dir::new("tracks");
        let t = tracks(&d, 4);
        let truth = oracle(&d, &t[3], 1.0);
        let mut w = window(&d.0.join("cache"), 1.0);
        let hold = park(&mut w);
        w.on_play(&t[0], SR);
        for p in &t[1..] {
            let s = Instant::now();
            w.on_play(p, SR);
            assert!(s.elapsed() < Duration::from_millis(500), "a track change waited for the worker");
            // The window's own state moved at once; only the job may wait.
            assert_eq!(w.current_path.as_deref(), Some(p.as_path()));
        }
        release(&mut w, &hold);
        let (ok, stale) = run(&mut w, playing, &truth);
        assert!(ok, "playing={playing}: never settled");
        assert_eq!(stale, 0, "playing={playing}: a stale picture was installed");
        assert!(*w.analyzer.pre_frames == truth, "playing={playing}: the newest track is not shown");
        assert_eq!(spawned(&w), 2, "playing={playing}: obsolete requests were started");
        assert_eq!(max_live(&w), 1, "playing={playing}: two workers were alive at once");
        if !playing {
            assert_eq!(drawn(&w), row_at_playhead(&truth, &w), "paused: the new picture is not drawn");
        }
    }
}

/// A setting changes (to one without a cache) while the analysis for the old
/// setting is alive. Nothing new starts by itself: the banner asks for
/// Re-analyze, the old setting's result is never installed, and Re-analyze —
/// available once the old worker has gone — runs exactly the newest setting.
#[test]
fn a_setting_change_while_busy_waits_for_re_analyze() {
    for playing in [false, true] {
        let d = Dir::new("setting");
        let t = tracks(&d, 1);
        let truth = oracle(&d, &t[0], 1.5);
        let mut w = window(&d.0.join("cache"), 1.0);
        let hold = park(&mut w);
        w.on_play(&t[0], SR);
        set_q(&mut w, 1.3);
        set_q(&mut w, 1.5);
        assert!(w.needs_reanalysis, "playing={playing}: no Re-analyze banner");
        release(&mut w, &hold);
        let (ok, stale) = run(&mut w, playing, &truth);
        assert!(ok, "playing={playing}: never settled");
        assert_eq!(stale, 0, "playing={playing}: the old setting's picture was installed");
        assert_eq!(spawned(&w), 1, "playing={playing}: a settings change started an analysis");
        assert!(w.analyzer.pre_frames.is_empty(), "playing={playing}: something was installed");
        assert!(w.reanalyze(), "playing={playing}: Re-analyze refused: {}", w.status_msg);
        let (ok, stale) = run(&mut w, playing, &truth);
        assert!(ok && stale == 0, "playing={playing}: Re-analyze did not settle cleanly");
        assert!(!w.needs_reanalysis, "playing={playing}: the banner outlived the analysis");
        assert!(*w.analyzer.pre_frames == truth, "playing={playing}: the newest setting is not shown");
        assert_eq!(spawned(&w), 2, "playing={playing}: Re-analyze did not run exactly once");
        assert_eq!(max_live(&w), 1, "playing={playing}: two workers were alive at once");
        if !playing {
            assert_eq!(drawn(&w), row_at_playhead(&truth, &w), "paused: the new picture is not drawn");
        }
    }
}

// ── A repeat keeps its work ────────────────────────────────────────────────

/// The same track at the same settings, started again (a loop, or a restart
/// from the top), while its analysis is alive: the work is kept, not
/// cancelled and restarted.
#[test]
fn a_repeat_keeps_the_analysis_in_flight() {
    let d = Dir::new("repeat");
    let t = tracks(&d, 1);
    let truth = oracle(&d, &t[0], 1.0);
    let mut w = window(&d.0.join("cache"), 1.0);
    let hold = park(&mut w);
    w.on_play(&t[0], SR);
    let abort = Arc::clone(&w.analyzer.abort_analysis);
    w.on_play(&t[0], SR);
    w.on_play(&t[0], SR);
    assert!(!abort.load(Ordering::SeqCst), "a repeat cancelled the analysis");
    release(&mut w, &hold);
    let (ok, stale) = run(&mut w, false, &truth);
    assert!(ok && stale == 0);
    assert!(*w.analyzer.pre_frames == truth);
    assert_eq!(spawned(&w), 1, "a repeat started the analysis again");
}

// ── Failure and cancellation give the slot back ───────────────────────────

/// A request that fails, then a valid one: the valid one runs.
#[test]
fn a_failure_releases_the_worker() {
    let d = Dir::new("fail");
    let t = tracks(&d, 1);
    let bad = d.0.join("bad.wav");
    std::fs::write(&bad, b"RIFF\0\0\0\0WAVEnot audio at all").unwrap();
    let truth = oracle(&d, &t[0], 1.0);
    let mut w = window(&d.0.join("cache"), 1.0);
    w.on_play(&bad, SR);
    let (ok, _) = run(&mut w, false, &cache::PreFrames::default());
    assert!(ok, "the failing request never settled");
    assert!(w.analyzer.pre_frames.is_empty());
    w.on_play(&t[0], SR);
    let (ok, stale) = run(&mut w, false, &truth);
    assert!(ok && stale == 0, "the request after a failure did not run cleanly");
    assert!(*w.analyzer.pre_frames == truth);
}

/// Stop while the analysis is alive, then a new track: the new track runs
/// once the cancelled worker has gone, never beside it, and nothing of the
/// cancelled one is shown.
#[test]
fn a_cancellation_releases_the_worker() {
    let d = Dir::new("cancel");
    let t = tracks(&d, 2);
    let truth = oracle(&d, &t[1], 1.0);
    let mut w = window(&d.0.join("cache"), 1.0);
    let hold = park(&mut w);
    w.on_play(&t[0], SR);
    w.on_stop();
    w.on_play(&t[1], SR);
    release(&mut w, &hold);
    let (ok, stale) = run(&mut w, false, &truth);
    assert!(ok, "never settled");
    assert_eq!(stale, 0, "the cancelled track's picture was shown");
    assert!(*w.analyzer.pre_frames == truth, "the track after a cancellation is not shown");
    assert_eq!(max_live(&w), 1, "the new worker ran beside the cancelled one");
}

// ── Stereo demand ─────────────────────────────────────────────────────────

/// "Analyse channels" pressed while the Mix analysis of the same track is
/// alive. The request must not vanish behind the busy worker: the channels
/// are produced once it has finished.
#[test]
fn a_channel_request_while_busy_is_not_lost() {
    let d = Dir::new("stereo");
    let t = tracks(&d, 1);
    let mut w = window(&d.0.join("cache"), 1.0);
    let hold = park(&mut w);
    w.on_play(&t[0], SR);
    w.analyzer.pre_stereo_request = Some(t[0].clone());
    w.analyzer.start_preprocess(t[0].clone());
    release(&mut w, &hold);
    // Settle fully, whatever was shown.
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut quiet = 0;
    while quiet < 3 && Instant::now() < deadline {
        frame(&mut w, false);
        quiet = if w.analyzer.analysis_idle() { quiet + 1 } else { 0 };
        std::thread::sleep(Duration::from_millis(2));
    }
    let key = cache_path_for(&w.analyzer.cache_dir(), &t[0], w.bar_count, w.fft_size, w.pad_factor,
                             w.overlap, &w.window_fn, w.min_freq, w.max_freq, &w.bar_mapping,
                             &w.interp_mode, w.analyzer.dsd_rate, &w.analyzer.aslt_cfg, w.analyzer.pre_fps);
    assert!(published_mix(&key), "no Mix cache");
    assert!(published_lr(&key), "the channel request was lost behind the busy worker");
}

// ── Gates ─────────────────────────────────────────────────────────────────

fn wait_reached(g: &Gate) {
    let deadline = Instant::now() + Duration::from_secs(120);
    while !g.reached() {
        assert!(Instant::now() < deadline, "the worker never reached the gate");
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Frames until nothing is running or waiting, whatever is shown.
fn settle(w: &mut SpectrumWindow) -> bool {
    run(w, false, &cache::PreFrames::default()).0
}

fn key(w: &SpectrumWindow, path: &Path) -> PathBuf {
    cache_path_for(&w.analyzer.cache_dir(), path, w.bar_count, w.fft_size, w.pad_factor,
                   w.overlap, &w.window_fn, w.min_freq, w.max_freq, &w.bar_mapping,
                   &w.interp_mode, w.analyzer.dsd_rate, &w.analyzer.aslt_cfg, w.analyzer.pre_fps)
}

fn live(w: &SpectrumWindow) -> usize { w.analyzer.lifecycle_probe.live.load(Ordering::SeqCst) }

// ── Cache controls against a cancelled writer ─────────────────────────────

#[derive(Clone, Copy, Debug)]
enum Control { ClearCache, ClearAll, Reanalyze }

#[derive(Clone, Copy, Debug)]
enum Cancel { Abort, Stop }

fn press(w: &mut SpectrumWindow, c: Control) -> bool {
    match c {
        Control::ClearCache => w.clear_cache(),
        Control::ClearAll => w.clear_all_caches(),
        Control::Reanalyze => w.reanalyze(),
    }
}

/// A worker cancelled after its last cancellation check still publishes its
/// cache. Each control is attempted in that window — after Abort or Stop,
/// before the rename — and then again once it is available. A clear must
/// not be undone by the late publication, and Re-analyze must analyse again
/// rather than reuse the file the cancelled writer has just published.
fn control_waits_for_a_cancelled_writer(control: Control) {
    for cancel in [Cancel::Abort, Cancel::Stop] {
        let ctx = format!("{cancel:?} then {control:?}");
        let d = Dir::new("controls");
        let t = tracks(&d, 1);
        let mut w = window(&d.0.join("cache"), 1.0);
        let g = Gate::closed();
        w.analyzer.lifecycle_probe.publish_gate = Some(g.clone());
        w.on_play(&t[0], SR);
        wait_reached(&g);
        let file = key(&w, &t[0]);
        assert!(!file.exists(), "{ctx}: setup: the cache was published before the gate");
        match cancel {
            Cancel::Abort => w.analyzer.abort_preprocess(),
            Cancel::Stop => w.on_stop(),
        }
        frame(&mut w, false);
        let early = press(&mut w, control);
        g.open();
        assert!(settle(&mut w), "{ctx}: never settled");
        assert!(file.exists(), "{ctx}: setup: the cancelled writer did not publish");
        if !early {
            assert!(press(&mut w, control), "{ctx}: the control stayed unavailable after the writer had gone");
        }
        assert!(settle(&mut w), "{ctx}: never settled after the control");
        match control {
            Control::ClearCache | Control::ClearAll => {
                assert!(!file.exists(), "{ctx}: the clear was undone by the cancelled writer's late publication");
            }
            Control::Reanalyze => {
                assert_eq!(spawned(&w), 2,
                    "{ctx}: Re-analyze reused the cancelled writer's cache instead of analysing");
                assert!(!w.analyzer.pre_frames.is_empty(), "{ctx}: Re-analyze installed nothing");
                assert!(file.exists(), "{ctx}: Re-analyze wrote no cache");
            }
        }
    }
}

#[test]
fn clear_cache_waits_for_a_cancelled_writer() { control_waits_for_a_cancelled_writer(Control::ClearCache) }

#[test]
fn clear_all_waits_for_a_cancelled_writer() { control_waits_for_a_cancelled_writer(Control::ClearAll) }

#[test]
fn reanalyze_waits_for_a_cancelled_writer() { control_waits_for_a_cancelled_writer(Control::Reanalyze) }

// ── The slot is held until the thread has exited ──────────────────────────

/// The worker has sent its result — a picture, or a failure — and its thread
/// has not exited. A request queued behind it must not start until it has:
/// only then is the old worker certainly done with the disk.
#[test]
fn the_next_worker_waits_for_the_last_to_exit() {
    for failing in [false, true] {
        let ctx = if failing { "after a failure" } else { "after a result" };
        let d = Dir::new("exit");
        let t = tracks(&d, 2);
        let bad = d.0.join("bad.wav");
        std::fs::write(&bad, b"RIFF\0\0\0\0WAVEnot audio at all").unwrap();
        let truth = oracle(&d, &t[1], 1.0);
        let mut w = window(&d.0.join("cache"), 1.0);
        let g = Gate::closed();
        w.analyzer.lifecycle_probe.exit_gate = Some(g.clone());
        w.on_play(if failing { &bad } else { &t[0] }, SR);
        wait_reached(&g);
        w.on_play(&t[1], SR);
        for _ in 0..50 {
            frame(&mut w, false);
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(spawned(&w), 1, "{ctx}: the next worker started before the last one's thread exited");
        assert_eq!(live(&w), 1, "{ctx}: setup: the parked worker is not alive");
        assert!(w.analyzer.analysis_active(), "{ctx}: the slot was given back while the thread lives");
        g.open();
        let (ok, stale) = run(&mut w, false, &truth);
        assert!(ok, "{ctx}: the queued request never completed");
        assert_eq!(stale, 0, "{ctx}: a stale picture was installed");
        assert!(*w.analyzer.pre_frames == truth, "{ctx}: the queued track is not shown");
        assert_eq!(spawned(&w), 2, "{ctx}: the queued request did not run exactly once");
        assert_eq!(max_live(&w), 1, "{ctx}: two workers were alive at once");
    }
}

/// A worker whose channel closes without a message — it panicked, say —
/// still owns the slot until its thread has exited.
#[test]
fn a_disconnected_worker_holds_the_slot_until_it_exits() {
    let d = Dir::new("disconnect");
    let t = tracks(&d, 2);
    let truth = oracle(&d, &t[1], 1.0);
    let mut w = window(&d.0.join("cache"), 1.0);
    let (tx, rx) = std::sync::mpsc::channel::<PreMessage>();
    let g = Gate::closed();
    let parked = g.clone();
    let thread = std::thread::spawn(move || {
        drop(tx);
        parked.pass();
    });
    w.analyzer.sample_rate = SR;
    w.analyzer.own_job_for_test(&t[0], rx, Some(thread));
    wait_reached(&g);
    w.on_play(&t[1], SR);
    for _ in 0..50 {
        frame(&mut w, false);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(spawned(&w), 0, "the next worker started beside a disconnected thread that is still alive");
    assert!(w.analyzer.analysis_active(), "the slot was given back while the thread lives");
    g.open();
    let (ok, stale) = run(&mut w, false, &truth);
    assert!(ok, "the queued request never completed");
    assert_eq!(stale, 0);
    assert!(*w.analyzer.pre_frames == truth, "the queued track is not shown");
    assert_eq!(spawned(&w), 1);
}

// ── Real-time, then Pre-process ───────────────────────────────────────────

/// A setting changes in Real-time, to one with no cache, while the analysis
/// for the old setting is in flight; then Pre-process is selected, paused,
/// without the track being played again. The approved rule: nothing is
/// analysed by the switch; the manual banner is raised, and Re-analyze runs
/// the setting chosen in Real-time, drawn at the paused playhead.
#[test]
fn a_realtime_miss_raises_the_banner_when_preprocess_returns() {
    let d = Dir::new("realtime");
    let t = tracks(&d, 1);
    let truth = oracle(&d, &t[0], 1.5);
    let mut w = window(&d.0.join("cache"), 1.0);
    let hold = park(&mut w);
    w.on_play(&t[0], SR);
    w.mode = SpectrumMode::RealTime;
    frame(&mut w, false);
    set_q(&mut w, 1.5);
    frame(&mut w, false);
    assert_eq!(spawned(&w), 1, "setup: a Real-time setting change started an analysis");
    w.mode = SpectrumMode::PreProcess;
    release(&mut w, &hold);
    let (ok, stale) = run(&mut w, false, &truth);
    assert!(ok, "never settled");
    assert_eq!(stale, 0, "the old setting's picture was installed");
    assert!(w.analyzer.pre_frames.is_empty(), "the switch installed something");
    assert!(w.needs_reanalysis, "the switch to Pre-process did not raise the banner");
    assert_eq!(spawned(&w), 1, "the switch to Pre-process started an analysis");
    assert!(w.reanalyze(), "Re-analyze refused: {}", w.status_msg);
    let (ok, stale) = run(&mut w, false, &truth);
    assert!(ok && stale == 0);
    assert!(*w.analyzer.pre_frames == truth, "the setting chosen in Real-time was never analysed");
    assert_eq!(spawned(&w), 2, "the new setting did not run exactly once");
    assert_eq!(max_live(&w), 1, "two workers were alive at once");
    assert_eq!(drawn(&w), row_at_playhead(&truth, &w), "paused: the new picture is not drawn");
}

/// The track information's spectral ceiling is measured from the installed
/// analysis. When a setting chosen in Real-time is analysed after returning to
/// Pre-process, the ceiling must describe the new analysis, not the old.
#[test]
fn the_spectral_ceiling_follows_the_replacement_analysis() {
    let d = Dir::new("ceiling");
    let t = tracks(&d, 1);
    let mut w = window(&d.0.join("cache"), 1.0);
    w.on_play(&t[0], SR);
    assert!(settle(&mut w));
    let old = w.spectral_ceiling.as_ref().map(|c| c.hz.to_bits());
    w.mode = SpectrumMode::RealTime;
    frame(&mut w, false);
    w.max_freq = 2_600.0;
    w.sync_params();
    w.try_load_or_flag_reanalysis();
    frame(&mut w, false);
    w.mode = SpectrumMode::PreProcess;
    frame(&mut w, false);
    assert!(w.reanalyze(), "Re-analyze refused: {}", w.status_msg);
    assert!(settle(&mut w));
    assert!(!w.analyzer.pre_frames.is_empty(), "the setting chosen in Real-time was never analysed");
    let want = compute_spectral_ceiling(
        &w.analyzer.pre_frames, w.bar_count, w.min_freq,
        w.max_freq.min(SR as f32 / 2.0), w.analyzer.aslt_cfg.scale,
    ).map(|c| c.hz.to_bits());
    assert_ne!(old, want, "setup: the two analyses have the same ceiling");
    assert_eq!(w.spectral_ceiling.as_ref().map(|c| c.hz.to_bits()), want,
               "the ceiling still describes the previous analysis");
}

// ── Explicit selection with demand pending ────────────────────────────────

/// A channel request is waiting behind the running Mix analysis when the
/// user chooses a legacy analysis for the track. The choice clears the
/// pending demand and retires the job: nothing else starts, nothing replaces
/// the choice, the slot is released, and the legacy cache's own channels are
/// still found beside it.
#[test]
fn an_explicit_choice_clears_pending_demand_and_retires_the_job() {
    for playing in [false, true] {
        let d = Dir::new("choice_pending");
        let t = tracks(&d, 1);
        let mut w = window(&d.0.join("cache"), 1.0);
        let (lid, legacy) = w.legacy_key_with(&t[0], &w.analyzer.aslt_cfg.clone());
        let rows = cache::PreFrames::from_analysis(&vec![vec![0.5f32; BARS]; 40]);
        cache::write_atomic(&legacy, &rows.to_v4()).unwrap();
        let pair = stereo::Stereo::new(rows.clone(), rows.clone()).unwrap();
        std::fs::write(stereo::path_for(&legacy), stereo::encode(lid, &rows, &pair)).unwrap();
        let hold = park(&mut w);
        w.on_play(&t[0], SR);
        w.analyzer.pre_stereo_request = Some(t[0].clone());
        w.analyzer.start_preprocess(t[0].clone());
        assert!(w.analyzer.pending_request.is_some(), "playing={playing}: setup: the channel request is not pending");
        w.accept_alternate(&legacy).expect("setup: legacy refused");
        assert!(w.analyzer.pending_request.is_none(), "playing={playing}: the pending demand survived the choice");
        release(&mut w, &hold);
        let (ok, stale) = run(&mut w, playing, &rows);
        assert!(ok, "playing={playing}: never settled");
        assert_eq!(stale, 0, "playing={playing}: a completion replaced the choice");
        assert!(*w.analyzer.pre_frames == rows, "playing={playing}: the choice is not shown");
        assert_eq!(spawned(&w), 1, "playing={playing}: the pending demand was started");
        assert_eq!(max_live(&w), 1);
        assert!(w.alt_in_use.as_ref().is_some_and(|c| c.legacy));
        assert!(w.analyzer.has_pre_channels(), "playing={playing}: the legacy cache's channels were not found");
    }
}

// ── Cache controls: truthful deletion ────────────────────────────────────
//
// Failures are injected with `cache::fault::fail_removal`, on this thread,
// because file permissions behave differently on every platform.

/// A window that has analysed `t[0]` and settled, with channels if asked.
fn analysed(d: &Dir, stereo_on: bool) -> (SpectrumWindow, Vec<PathBuf>) {
    let t = tracks(d, 1);
    let mut w = window(&d.0.join("cache"), 1.0);
    w.analyzer.pre_stereo_enabled = stereo_on;
    w.on_play(&t[0], SR);
    assert!(settle(&mut w), "setup: the first analysis never settled");
    assert!(!w.analyzer.pre_frames.is_empty(), "setup: nothing installed");
    (w, t)
}

/// The generation files a unit's head names now: (Mix, channels).
fn unit_files(w: &SpectrumWindow, track: &Path) -> (PathBuf, Option<PathBuf>) {
    head_files(&key(w, track)).expect("setup: no analysis on disk")
}

fn store_dir_of(w: &SpectrumWindow) -> PathBuf {
    identity::store_dir(&w.analyzer.cache_dir())
}

/// Make removals of the payload file named like `target` fail in the store of
/// `w` while the returned guard lives.
fn undeletable(w: &SpectrumWindow, target: &Path) -> store::seam::Armed {
    let name = target.file_name().unwrap().to_owned();
    store::seam::fault(&store_dir_of(w), move |point, p| {
        (point == "payload.remove" && p.file_name() == Some(name.as_os_str())).then_some(std::io::ErrorKind::PermissionDenied)
    })
}

/// Clear Cache removes the track's analysis: its head, Mix and channels.
#[test]
fn clear_cache_removes_the_mix_and_its_channels() {
    let d = Dir::new("unit_clear");
    let (mut w, t) = analysed(&d, true);
    let unit = key(&w, &t[0]);
    let (mix, lr) = unit_files(&w, &t[0]);
    let lr = lr.expect("setup: no channels");
    assert!(w.clear_cache(), "Clear Cache failed: {}", w.status_msg);
    assert!(!unit.exists() && !mix.exists() && !lr.exists(), "the analysis outlived Clear Cache");
    assert_eq!(w.status_msg, "Cache cleared.");
    assert!(w.analyzer.pre_frames.is_empty(), "the cleared picture is still shown");
}

/// Re-analyze deletes nothing first: the old analysis stays what the head
/// names until the new one commits; then the superseded generation is
/// collected, and the Mix is computed again rather than carried over.
#[test]
fn reanalyze_replaces_the_analysis_when_the_new_one_commits() {
    let d = Dir::new("unit_reanalyze");
    let (mut w, t) = analysed(&d, true);
    let unit = key(&w, &t[0]);
    let (old_mix, old_lr) = unit_files(&w, &t[0]);
    let old_lr = old_lr.expect("setup: no channels");
    let hold = park(&mut w);
    assert!(w.reanalyze(), "Re-analyze failed: {}", w.status_msg);
    assert!(unit.exists() && old_mix.exists() && old_lr.exists(), "Re-analyze removed the analysis before a new one existed");
    assert_eq!(head_files(&unit).unwrap().0, old_mix, "the head moved before the new analysis committed");
    release(&mut w, &hold);
    assert!(settle(&mut w));
    assert_eq!(spawned(&w), 2, "Re-analyze did not analyse");
    let (new_mix, new_lr) = unit_files(&w, &t[0]);
    assert_ne!(new_mix, old_mix, "no new generation");
    assert!(new_lr.is_some(), "the new analysis has no channels");
    assert!(!old_mix.exists() && !old_lr.exists(), "the superseded generation was not collected");
    assert!(!w.analyzer.pre_frames.is_empty(), "the new analysis is not shown");
}

/// A deletion that fails is reported as a failure. The head is gone, so the
/// analysis is no longer found; the generation left behind keeps its pin file
/// and goes on the next attempt.
#[test]
fn a_failed_clear_is_reported() {
    let d = Dir::new("clear_fail");
    let (mut w, t) = analysed(&d, true);
    let unit = key(&w, &t[0]);
    let (mix, _) = unit_files(&w, &t[0]);
    let fault = undeletable(&w, &mix);
    assert!(!w.clear_cache(), "a failed deletion was reported as success");
    assert!(w.status_msg.starts_with("Could not clear this analysis completely"), "status {:?}", w.status_msg);
    assert!(w.status_msg.contains("1 could not be deleted"), "status {:?}", w.status_msg);
    assert!(mix.exists(), "setup: the undeletable Mix is gone");
    assert!(!unit.exists(), "the head still names a generation that is being removed");
    drop(fault);
    assert!(w.clear_cache(), "the retry failed: {}", w.status_msg);
    assert!(!mix.exists(), "the retry left the generation");
}

/// Clear All says exactly what it could not do — here, list the folder —
/// never "All caches cleared" when some remain, and keeps the picture.
#[test]
fn clear_all_reports_a_listing_failure() {
    let d = Dir::new("clear_all");
    let t = tracks(&d, 2);
    let mut w = window(&d.0.join("cache"), 1.0);
    for p in &t {
        w.on_play(p, SR);
        assert!(settle(&mut w));
    }
    let (k0, k1) = (key(&w, &t[0]), key(&w, &t[1]));
    assert!(k0.exists() && k1.exists(), "setup: two analyses expected");
    let _f = store::seam::fault(&store_dir_of(&w), |point, _| (point == "dir.list").then_some(std::io::ErrorKind::PermissionDenied));
    assert!(!w.clear_all_caches(), "an unlisted folder was reported cleared");
    assert!(w.status_msg.starts_with("Could not list"), "status {:?}", w.status_msg);
    assert!(k0.exists() && k1.exists());
    assert!(!w.analyzer.pre_frames.is_empty(), "the picture was dropped although nothing was cleared");
}

/// One generation of two cannot be deleted. The current track is t[1].
#[test]
fn clear_all_reports_a_partial_result() {
    let d = Dir::new("clear_all_partial");
    let t = tracks(&d, 2);
    let mut w = window(&d.0.join("cache"), 1.0);
    for p in &t {
        w.on_play(p, SR);
        assert!(settle(&mut w));
    }
    let (m0, _) = unit_files(&w, &t[0]);
    let (m1, _) = unit_files(&w, &t[1]);
    let _f = undeletable(&w, &m0);
    assert!(!w.clear_all_caches(), "a partial Clear All was reported as complete");
    assert!(!w.status_msg.contains("All caches cleared"), "status {:?}", w.status_msg);
    assert!(w.status_msg.contains("1 could not be deleted"), "status {:?}", w.status_msg);
    assert!(m0.exists(), "setup: the undeletable file is gone");
    assert!(!m1.exists(), "a deletable analysis was left");
    assert!(w.analyzer.pre_frames.is_empty(), "the current track's analysis is gone but still shown");
}

/// An analysis something still reads (here: a holder of the installed
/// generation, as a lazy channel read is) is pinned: Clear All keeps it and
/// says so, and clears it once nothing holds it (D8).
#[test]
fn clear_all_keeps_and_reports_an_analysis_in_use() {
    let d = Dir::new("clear_all_pinned");
    let (mut w, t) = analysed(&d, false);
    let unit = key(&w, &t[0]);
    let held = w.analyzer.pre_generation_installed().cloned().expect("setup: no store analysis shown");
    assert!(!w.clear_all_caches(), "an analysis in use was reported cleared");
    assert!(w.status_msg.contains("in use and kept"), "status {:?}", w.status_msg);
    assert!(unit.exists() && held.mix_path().exists(), "a pinned analysis was deleted");
    drop(held);
    assert!(w.clear_all_caches(), "status {:?}", w.status_msg);
    assert!(!unit.exists());
}

/// Re-analyze discards the Track Info ceiling with the frames it was
/// measured from, and measures it again from the replacement. The file is
/// replaced in between, as a user would before forcing a re-analysis, so the
/// two ceilings differ.
#[test]
fn reanalyze_remeasures_the_spectral_ceiling() {
    let d = Dir::new("reanalyze_ceiling");
    let src = d.0.join("song.wav");
    write_wav(&src, &[300.0, 2100.0], 7);
    let mut w = window(&d.0.join("cache"), 1.0);
    w.on_play(&src, SR);
    assert!(settle(&mut w));
    let old = w.spectral_ceiling.as_ref().map(|c| c.hz.to_bits());
    write_wav_noise(&src, &[300.0], 7, 0.0);
    assert!(w.reanalyze(), "Re-analyze failed: {}", w.status_msg);
    assert!(settle(&mut w));
    assert_eq!(spawned(&w), 2, "setup: Re-analyze did not analyse");
    let want = compute_spectral_ceiling(
        &w.analyzer.pre_frames, w.bar_count, w.min_freq,
        w.max_freq.min(SR as f32 / 2.0), w.analyzer.aslt_cfg.scale,
    ).map(|c| c.hz.to_bits());
    assert_ne!(old, want, "setup: the two analyses have the same ceiling");
    assert_eq!(w.spectral_ceiling.as_ref().map(|c| c.hz.to_bits()), want,
               "Track Info's ceiling still describes the analysis Re-analyze replaced");
}

/// Clear All with no cache folder at all. Nothing is left on disk, so the
/// installed picture goes, exactly as when Clear All deletes the current
/// track's Mix cache.
#[test]
fn clear_all_without_a_cache_folder_drops_the_picture() {
    let d = Dir::new("clear_all_absent");
    let (mut w, _t) = analysed(&d, false);
    std::fs::remove_dir_all(w.analyzer.cache_dir()).unwrap();
    assert!(!w.analyzer.pre_frames.is_empty(), "setup: no picture installed");
    assert!(w.clear_all_caches(), "status {:?}", w.status_msg);
    assert_eq!(w.status_msg, "All caches cleared.");
    assert!(w.analyzer.pre_frames.is_empty(), "a picture whose cache is gone is still shown");
    assert!(w.needs_reanalysis, "the missing cache is not flagged for re-analysis");
}

// ── A queued request keeps the settings it was made at ───────────────────
//
// Choosing a track authorises its analysis at the settings in force at that
// moment, and nothing else does. A request queued behind a running worker
// carries that descriptor; when the slot frees it starts only if the
// descriptor is still the one selected. A settings edit made meanwhile is a
// settings edit like any other: a miss raises the manual banner.

/// What the analyser launched, in order: descriptor, rate, and whether the
/// worker produced channels.
fn launched(w: &SpectrumWindow) -> Vec<(AnalysisSettings, u32, bool)> {
    w.analyzer.lifecycle_probe.launched.lock().unwrap().clone()
}

fn tracks_of(l: &[(AnalysisSettings, u32, bool)]) -> Vec<PathBuf> {
    l.iter().map(|(s, _, _)| s.track.clone()).collect()
}

/// A's worker is alive; B is chosen at S1 and queues; B's setting is moved to
/// an uncached S2 through the slider's path; A exits. B must not be analysed
/// at S2 (that needs Re-analyze), nor at S1 (no longer selected).
#[test]
fn a_queued_track_is_not_analysed_at_settings_chosen_after_it_queued() {
    for playing in [false, true] {
        let d = Dir::new("queued_miss");
        let t = tracks(&d, 2);
        let truth = oracle(&d, &t[1], 1.5);
        let mut w = window(&d.0.join("cache"), 1.0);
        let hold = park(&mut w);
        w.on_play(&t[0], SR);
        w.on_play(&t[1], SR);
        assert!(w.analyzer.pending_request.is_some(), "playing={playing}: setup: B is not queued");
        set_q(&mut w, 1.5);
        assert!(w.needs_reanalysis, "playing={playing}: setup: no banner for the uncached setting");
        release(&mut w, &hold);
        let (ok, stale) = run(&mut w, playing, &truth);
        assert!(ok, "playing={playing}: never settled");
        assert_eq!(stale, 0, "playing={playing}: a picture was installed");
        let l = launched(&w);
        assert_eq!(tracks_of(&l), vec![t[0].clone()], "playing={playing}: launched {:?}", tracks_of(&l));
        assert_eq!(l[0].0, w.settings_with(&t[0], &cfg(1.0)), "playing={playing}: A's launch");
        assert!(w.analyzer.pre_frames.is_empty(), "playing={playing}: something is shown");
        assert!(w.needs_reanalysis, "playing={playing}: the manual banner was cleared");
        assert!(!w.key_with(&t[1], &cfg(1.5)).1.exists(), "playing={playing}: B was analysed at S2 without Re-analyze");
        assert!(!w.key_with(&t[1], &cfg(1.0)).1.exists(), "playing={playing}: B was analysed at S1 after S1 was left");
        // The manual step is still what starts it, and it starts S2.
        assert!(w.reanalyze(), "playing={playing}: Re-analyze refused: {}", w.status_msg);
        let (ok, stale) = run(&mut w, playing, &truth);
        assert!(ok && stale == 0, "playing={playing}: Re-analyze did not settle cleanly");
        let l = launched(&w);
        assert_eq!(l.len(), 2, "playing={playing}: launched {:?}", tracks_of(&l));
        assert_eq!(l[1].0, w.settings_with(&t[1], &cfg(1.5)), "playing={playing}: Re-analyze ran the wrong descriptor");
        assert!(*w.analyzer.pre_frames == truth, "playing={playing}: S2 is not shown");
        assert_eq!(w.analyzer.pre_cache_path(), Some(w.key_with(&t[1], &cfg(1.5)).1.as_path()));
        assert!(!w.needs_reanalysis, "playing={playing}: the banner outlived the analysis");
        assert_eq!(max_live(&w), 1, "playing={playing}: two workers were alive at once");
    }
}

/// The same, but the setting is moved back to S1 before A exits. B was
/// authorised at S1 and S1 is selected again: B runs at S1, by itself, and
/// answers the banner that the round trip raised.
#[test]
fn a_queued_track_runs_when_its_settings_are_restored_before_the_slot_frees() {
    for playing in [false, true] {
        let d = Dir::new("queued_back");
        let t = tracks(&d, 2);
        let truth = oracle(&d, &t[1], 1.0);
        let mut w = window(&d.0.join("cache"), 1.0);
        let hold = park(&mut w);
        w.on_play(&t[0], SR);
        w.on_play(&t[1], SR);
        set_q(&mut w, 1.5);
        set_q(&mut w, 1.0);
        release(&mut w, &hold);
        let (ok, stale) = run(&mut w, playing, &truth);
        assert!(ok, "playing={playing}: never settled");
        assert_eq!(stale, 0, "playing={playing}: a stale picture was installed");
        let l = launched(&w);
        assert_eq!(tracks_of(&l), vec![t[0].clone(), t[1].clone()], "playing={playing}");
        assert_eq!(l[1].0, w.settings_with(&t[1], &cfg(1.0)), "playing={playing}: B ran at other settings");
        assert!(*w.analyzer.pre_frames == truth, "playing={playing}: B is not shown");
        assert_eq!(w.analyzer.pre_cache_path(), Some(w.key_with(&t[1], &cfg(1.0)).1.as_path()));
        assert!(!w.needs_reanalysis, "playing={playing}: the banner outlived the analysis that answered it");
        assert!(!w.key_with(&t[1], &cfg(1.5)).1.exists(), "playing={playing}: S2 was analysed");
        assert_eq!(max_live(&w), 1);
        if !playing {
            assert_eq!(drawn(&w), row_at_playhead(&truth, &w), "paused: B is not drawn");
        }
    }
}

/// B is queued at S1 when the setting moves to S2, for which B has a cache.
/// The hit is shown at once; when A exits nothing is started for B and the
/// hit stays.
#[test]
fn an_exact_hit_while_a_request_is_queued_is_shown_and_the_request_lapses() {
    for playing in [false, true] {
        let d = Dir::new("queued_hit");
        let t = tracks(&d, 2);
        let mut w = window(&d.0.join("cache"), 1.5);
        w.on_play(&t[1], SR);
        assert!(settle(&mut w), "playing={playing}: setup: B at S2 never settled");
        let b_s2 = (*w.analyzer.pre_frames).clone();
        assert!(!b_s2.is_empty(), "playing={playing}: setup: no B at S2");
        set_q(&mut w, 1.0);
        let hold = park(&mut w);
        w.on_play(&t[0], SR);
        w.on_play(&t[1], SR);
        assert!(w.analyzer.pending_request.is_some(), "playing={playing}: setup: B is not queued");
        set_q(&mut w, 1.5);
        assert!(*w.analyzer.pre_frames == b_s2, "playing={playing}: the hit was not installed");
        assert!(!w.needs_reanalysis, "playing={playing}: a hit raised the banner");
        release(&mut w, &hold);
        let (ok, stale) = run(&mut w, playing, &b_s2);
        assert!(ok, "playing={playing}: never settled");
        assert_eq!(stale, 0, "playing={playing}: the hit was replaced");
        assert!(*w.analyzer.pre_frames == b_s2, "playing={playing}: the hit is gone");
        assert_eq!(w.analyzer.pre_cache_path(), Some(w.key_with(&t[1], &cfg(1.5)).1.as_path()));
        let l = launched(&w);
        assert_eq!(tracks_of(&l), vec![t[1].clone(), t[0].clone()], "playing={playing}: B was started again");
        assert_eq!(l[1].0, w.settings_with(&t[0], &cfg(1.0)), "playing={playing}: A's launch");
        assert!(!w.key_with(&t[1], &cfg(1.0)).1.exists(), "playing={playing}: B at S1 ran");
        assert!(!w.needs_reanalysis, "playing={playing}");
    }
}

/// B is queued behind A when a legacy analysis of B is chosen. The choice
/// retires A and clears B's request: nothing else starts, and the choice
/// stays on screen.
#[test]
fn an_explicit_choice_for_a_queued_track_clears_it_and_retires_the_job() {
    for playing in [false, true] {
        let d = Dir::new("queued_choice");
        let t = tracks(&d, 2);
        let mut w = window(&d.0.join("cache"), 1.0);
        let (lid, legacy) = w.legacy_key_with(&t[1], &w.analyzer.aslt_cfg.clone());
        let rows = cache::PreFrames::from_analysis(&vec![vec![0.25f32; BARS]; 40]);
        cache::write_atomic(&legacy, &rows.to_v4()).unwrap();
        let pair = stereo::Stereo::new(rows.clone(), rows.clone()).unwrap();
        std::fs::write(stereo::path_for(&legacy), stereo::encode(lid, &rows, &pair)).unwrap();
        let hold = park(&mut w);
        w.on_play(&t[0], SR);
        w.on_play(&t[1], SR);
        assert!(w.analyzer.pending_request.is_some(), "playing={playing}: setup: B is not queued");
        w.accept_alternate(&legacy).expect("setup: legacy refused");
        assert!(w.analyzer.pending_request.is_none(), "playing={playing}: B's request survived the choice");
        release(&mut w, &hold);
        let (ok, stale) = run(&mut w, playing, &rows);
        assert!(ok, "playing={playing}: never settled");
        assert_eq!(stale, 0, "playing={playing}: a completion replaced the choice");
        assert!(*w.analyzer.pre_frames == rows, "playing={playing}: the choice is not shown");
        assert_eq!(w.analyzer.pre_cache_path(), Some(legacy.as_path()));
        assert!(w.alt_in_use.as_ref().is_some_and(|c| c.legacy), "playing={playing}");
        assert_eq!(tracks_of(&launched(&w)), vec![t[0].clone()], "playing={playing}: B was started");
        assert!(!w.key_with(&t[1], &cfg(1.0)).1.exists(), "playing={playing}: B was analysed");
        assert_eq!(max_live(&w), 1);
    }
}

/// B queued at S1, the setting moved to an uncached S2, then C chosen.
/// Choosing C authorises C at S2: it runs by itself. B runs at nothing.
#[test]
fn a_track_chosen_after_a_settings_change_is_analysed_at_the_new_settings() {
    let d = Dir::new("queued_track");
    let t = tracks(&d, 3);
    let truth = oracle(&d, &t[2], 1.5);
    let mut w = window(&d.0.join("cache"), 1.0);
    let hold = park(&mut w);
    w.on_play(&t[0], SR);
    w.on_play(&t[1], SR);
    set_q(&mut w, 1.5);
    w.on_play(&t[2], SR);
    release(&mut w, &hold);
    let (ok, stale) = run(&mut w, false, &truth);
    assert!(ok && stale == 0, "never settled cleanly");
    let l = launched(&w);
    assert_eq!(tracks_of(&l), vec![t[0].clone(), t[2].clone()]);
    assert_eq!(l[1].0, w.settings_with(&t[2], &cfg(1.5)), "C ran at other settings");
    assert!(*w.analyzer.pre_frames == truth, "C is not shown");
    assert!(!w.needs_reanalysis);
    for q in [1.0, 1.5] {
        assert!(!w.key_with(&t[1], &cfg(q)).1.exists(), "B was analysed at q {q}");
    }
}

/// "Analyse channels" for B while B is queued behind A. Untouched, the
/// request runs at S1 with channels; with the setting moved to an uncached S2
/// first, it lapses like any queued request, channels and all.
#[test]
fn a_queued_channel_request_keeps_its_settings_or_lapses_with_them() {
    for change in [false, true] {
        let d = Dir::new("queued_stereo");
        let t = tracks(&d, 2);
        let mut w = window(&d.0.join("cache"), 1.0);
        let hold = park(&mut w);
        w.on_play(&t[0], SR);
        w.on_play(&t[1], SR);
        w.analyzer.pre_stereo_request = Some(t[1].clone());
        w.analyzer.start_preprocess(t[1].clone());
        assert!(w.analyzer.pending_request.as_ref().is_some_and(|p| p.explicit), "change={change}: setup");
        if change {
            set_q(&mut w, 1.5);
        }
        release(&mut w, &hold);
        assert!(settle(&mut w), "change={change}: never settled");
        let l = launched(&w);
        let s1 = w.key_with(&t[1], &cfg(1.0)).1;
        if change {
            assert_eq!(tracks_of(&l), vec![t[0].clone()], "change={change}: B was started");
            assert!(!published_mix(&s1), "change={change}: B ran at S1");
            assert!(!w.key_with(&t[1], &cfg(1.5)).1.exists(), "change={change}: B ran at S2");
            assert!(w.needs_reanalysis, "change={change}: the banner was cleared");
        } else {
            assert_eq!(tracks_of(&l), vec![t[0].clone(), t[1].clone()], "change={change}");
            assert_eq!(l[1].0, w.settings_with(&t[1], &cfg(1.0)), "change={change}: B ran at other settings");
            assert!(l[1].2, "change={change}: the channel demand was lost");
            assert!(published_lr(&s1), "change={change}: no channels for B");
            assert_eq!(w.analyzer.pre_cache_path(), Some(s1.as_path()));
        }
        assert_eq!(max_live(&w), 1);
    }
}

// ── The analysis cache store through production entry points ─────────────
#[path = "cache_v2_tests.rs"]
mod cache_v2;

// ── The whole-track view through production entry points ─────────────────
#[path = "wholetrack_int_tests.rs"]
mod wholetrack_int;

// ── The whole-track view's component cost (ignored benchmark) ─────────────
#[path = "wholetrack_bench.rs"]
mod wholetrack_bench;

//! High-accuracy GPU through the app's own analysis seam: `request_preprocess`,
//! the worker (`preprocess_file_mode`), `finish_job`, `service_preprocess` and
//! the cache store, as a track is played. The HP pass itself is injected per
//! analyser (`hp::State::hook`), so every failure path is deterministic and
//! needs no device; the real engine has its own ignored checks at the end.
//!
//! Isolation as in `lifecycle_tests`: isolated windows, a private cache folder
//! per test, generated WAV inputs; nothing under the home profile is read or
//! written.

use super::*;
use std::time::{Duration, Instant};

const SR: u32 = 44_100;
const BARS: usize = 1024;

struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "moosik_hp_seam_{tag}_{}_{:x}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0),
        ));
        std::fs::create_dir_all(&root).unwrap();
        Dir(root)
    }

    /// A short stereo WAV at 44.1 kHz whose channels differ.
    fn track(&self, name: &str, hz: f32) -> PathBuf {
        let path = self.0.join(name);
        let n = SR as usize * 2;
        let mut data = Vec::with_capacity(n * 4);
        for i in 0..n {
            let t = i as f32 / SR as f32;
            let tone = (std::f32::consts::TAU * hz * t).sin();
            for s in [0.3 * tone, 0.2 * tone] { data.extend(((s * 32_767.0) as i16).to_le_bytes()) }
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
        std::fs::write(&path, out).unwrap();
        path
    }
}

impl Drop for Dir {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

/// A window at the validated descriptor, Superlet, on a private cache, with
/// High-accuracy GPU preferred unless `hp` is false.
fn window(cache: &Path, hp: bool) -> SpectrumWindow {
    let mut w = SpectrumWindow::new_isolated();
    w.mode = SpectrumMode::PreProcess;
    w.analyzer.cache_dir_override = Some(cache.to_path_buf());
    w.analyzer.cache_budget_gb = 0.0;
    w.analyzer.pre_stereo_enabled = false;
    w.bar_count = BARS;
    w.analyzer.bar_count = BARS;
    w.bar_mapping = BarMappingMode::Superlet;
    w.analyzer.bar_mapping = BarMappingMode::Superlet;
    w.min_freq = 20.0;
    w.max_freq = 24_000.0;
    w.analyzer.min_freq = 20.0;
    w.analyzer.max_freq = 24_000.0;
    w.analyzer.aslt_preset = None;
    w.analyzer.aslt_cfg = aslt::hp::validated_cfg();
    w.analyzer.pre_fps = 180.0;
    w.analyzer.set_hp_enabled(hp);
    w
}

fn frame(w: &mut SpectrumWindow) {
    w.last_fft_time = Some(Instant::now() - Duration::from_millis(20));
    w.tick(0.5, false);
}

/// Frames until nothing is running or waiting.
fn run(w: &mut SpectrumWindow) {
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut quiet = 0;
    loop {
        frame(w);
        quiet = if w.analyzer.analysis_idle() { quiet + 1 } else { 0 };
        if quiet >= 3 { return }
        assert!(Instant::now() < deadline, "the analysis never settled");
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Frames until `flag` is set.
fn until(w: &mut SpectrumWindow, flag: &AtomicBool) {
    let deadline = Instant::now() + Duration::from_secs(120);
    while !flag.load(Ordering::SeqCst) {
        frame(w);
        assert!(Instant::now() < deadline, "never reached");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Codes in the real engine's shape, a different value per plane.
fn codes(l: &[f32], sr: u32) -> ([Vec<u16>; 3], usize) {
    let frames = aslt::frame_count(l.len(), aslt::hop_for_fps(sr, 180.0));
    let n = frames * BARS;
    ([vec![1000; n], vec![1100; n], vec![1200; n]], frames)
}

fn hook(f: impl Fn(&[f32], &[f32], u32, &AtomicBool, &(dyn Fn(usize) + Sync))
    -> Result<([Vec<u16>; 3], usize), aslt::hp::HpError> + Send + Sync + 'static) -> hp::Hook
{
    Arc::new(f)
}

/// The numerics of every analysis the window launched, in order, by track name.
fn launched(w: &SpectrumWindow) -> Vec<(String, Numerics)> {
    w.analyzer.lifecycle_probe.launched.lock().unwrap().iter()
        .map(|(s, _, _)| (s.track.file_name().unwrap().to_string_lossy().into_owned(), s.numerics()))
        .collect()
}

fn kinds(w: &SpectrumWindow) -> Vec<&'static str> {
    w.analyzer.lifecycle_probe.outcomes.lock().unwrap().iter().map(|o| o.kind).collect()
}

/// What the installed analysis says it was computed with.
fn shown(w: &SpectrumWindow) -> Option<Numerics> {
    if w.analyzer.pre_frames.is_empty() { return None; }
    let record = w.analyzer.pre_record.as_deref()?;
    Some(if record.contains(";num=gpu-hp1") { Numerics::HighAccuracyGpu } else { Numerics::Standard })
}

use Numerics::{HighAccuracyGpu as Hp, Standard};

/// An HP result is published under its own identity and is the exact hit
/// from then on. A Standard window never takes it for a Standard result.
#[test]
fn an_hp_result_is_its_own_exact_hit_and_never_a_standard_one() {
    let d = Dir::new("hit");
    let t = d.track("a.wav", 440.0);
    let cache = d.0.join("cache");
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = |calls: &Arc<AtomicUsize>| {
        let calls = Arc::clone(calls);
        hook(move |l, _, sr, _, _| { calls.fetch_add(1, Ordering::SeqCst); Ok(codes(l, sr)) })
    };

    let mut w = window(&cache, true);
    w.analyzer.hp.hook = Some(counted(&calls));
    w.on_play(&t, SR);
    run(&mut w);
    assert_eq!(launched(&w), [("a.wav".into(), Hp)]);
    assert_eq!(shown(&w), Some(Hp));
    assert_eq!(w.analyzer.pre_frames.row(0).map(|r| r[0]), Some(1000), "the injected Mix was installed");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(w.analyzer.hp_label().is_some_and(|l| l.starts_with("High-accuracy GPU numerics")));

    // Played again in a new window: the exact hit, nothing launched.
    let mut again = window(&cache, true);
    again.analyzer.hp.hook = Some(counted(&calls));
    again.on_play(&t, SR);
    run(&mut again);
    assert!(launched(&again).is_empty(), "an exact HP hit launched an analysis");
    assert_eq!(shown(&again), Some(Hp));
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Standard preferred: the HP result is not Standard's, so Standard runs.
    let mut standard = window(&cache, false);
    standard.on_play(&t, SR);
    run(&mut standard);
    assert_eq!(launched(&standard), [("a.wav".into(), Standard)]);
    assert_eq!(shown(&standard), Some(Standard));
}

/// Policy (a): with HP preferred and only a Standard result on disk, that
/// result is shown, labelled Standard, and nothing is launched. An explicit
/// reanalysis then runs HP, which is the exact hit from then on.
#[test]
fn an_existing_standard_result_is_shown_as_standard_until_hp_is_asked_for() {
    let d = Dir::new("policy_a");
    let t = d.track("a.wav", 440.0);
    let cache = d.0.join("cache");
    let mut standard = window(&cache, false);
    standard.on_play(&t, SR);
    run(&mut standard);
    assert_eq!(shown(&standard), Some(Standard));

    let calls = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&calls);
    let mut w = window(&cache, true);
    w.analyzer.hp.hook = Some(hook(move |l, _, sr, _, _| { c.fetch_add(1, Ordering::SeqCst); Ok(codes(l, sr)) }));
    w.on_play(&t, SR);
    run(&mut w);
    assert!(launched(&w).is_empty(), "policy (a) launched an analysis");
    assert_eq!(shown(&w), Some(Standard));
    assert!(w.analyzer.hp_label().is_some_and(|l| l.starts_with("Standard numerics")), "{:?}", w.analyzer.hp_label());
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    w.analyzer.request_hp_reanalysis(t.clone());
    run(&mut w);
    assert_eq!(launched(&w), [("a.wav".into(), Hp)]);
    assert_eq!(shown(&w), Some(Hp));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// A descriptor outside the validated one is refused on the worker before
/// the decode, and the request is retried once, under Standard.
#[test]
fn a_refused_request_is_retried_once_under_standard() {
    let d = Dir::new("refused");
    let t = d.track("a.wav", 440.0);
    let calls = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&calls);
    let mut w = window(&d.0.join("cache"), true);
    w.bar_count = 128;
    w.analyzer.bar_count = 128;
    w.analyzer.hp.hook = Some(hook(move |l, _, sr, _, _| { c.fetch_add(1, Ordering::SeqCst); Ok(codes(l, sr)) }));
    w.on_play(&t, SR);
    run(&mut w);
    assert_eq!(launched(&w), [("a.wav".into(), Hp), ("a.wav".into(), Standard)]);
    assert_eq!(kinds(&w), ["hp_unavailable", "done"]);
    assert_eq!(shown(&w), Some(Standard));
    assert_eq!(calls.load(Ordering::SeqCst), 0, "a refused request reached the GPU pass");
    assert!(w.analyzer.hp_label().is_some_and(|l| l.contains("256 bars")), "{:?}", w.analyzer.hp_label());
}

/// A device failure: one Standard retry under Standard identity, and no HP
/// for any track for the rest of the session.
#[test]
fn a_device_failure_is_retried_once_under_standard_and_ends_hp_for_the_session() {
    let d = Dir::new("device");
    let (a, b) = (d.track("a.wav", 440.0), d.track("b.wav", 660.0));
    let calls = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&calls);
    let mut w = window(&d.0.join("cache"), true);
    w.analyzer.hp.hook = Some(hook(move |_, _, _, _, _| {
        c.fetch_add(1, Ordering::SeqCst);
        Err(aslt::hp::HpError::Device("injected launch failure".into()))
    }));
    w.on_play(&a, SR);
    run(&mut w);
    assert_eq!(launched(&w), [("a.wav".into(), Hp), ("a.wav".into(), Standard)]);
    assert_eq!(shown(&w), Some(Standard));
    assert!(w.analyzer.hp_label().is_some_and(|l| l.contains("injected launch failure")), "{:?}", w.analyzer.hp_label());
    w.on_play(&b, SR);
    run(&mut w);
    assert_eq!(launched(&w)[2..], [("b.wav".into(), Standard)], "HP was tried again after a device failure");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// Cancelled while the pass also failed: nothing restarts, nothing is shown,
/// and the device failure still ends HP for the session.
#[test]
fn a_cancelled_pass_never_restarts_even_when_it_also_failed() {
    let d = Dir::new("cancel");
    let (a, b) = (d.track("a.wav", 440.0), d.track("b.wav", 660.0));
    let mut w = window(&d.0.join("cache"), true);
    w.analyzer.hp.hook = Some(hook(|_, _, _, abort, _| {
        abort.store(true, Ordering::SeqCst);
        Err(aslt::hp::HpError::Device("injected failure during the cancellation".into()))
    }));
    w.on_play(&a, SR);
    run(&mut w);
    assert_eq!(launched(&w), [("a.wav".into(), Hp)], "a cancelled pass was restarted");
    assert_eq!(kinds(&w), ["aborted"]);
    assert_eq!(shown(&w), None);
    w.on_play(&b, SR);
    run(&mut w);
    assert_eq!(launched(&w)[1..], [("b.wav".into(), Standard)]);
}

/// Superseded while the HP pass runs: its failure gets no retry, and the
/// newer request is the one analysed.
#[test]
fn a_superseded_request_is_never_retried() {
    let d = Dir::new("superseded");
    let (a, b) = (d.track("a.wav", 440.0), d.track("b.wav", 660.0));
    let (inside, release) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let (i, r) = (Arc::clone(&inside), Arc::clone(&release));
    let calls = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&calls);
    let mut w = window(&d.0.join("cache"), true);
    w.analyzer.hp.hook = Some(hook(move |l, _, sr, _, _| {
        if c.fetch_add(1, Ordering::SeqCst) == 0 {
            i.store(true, Ordering::SeqCst);
            while !r.load(Ordering::SeqCst) { std::thread::sleep(Duration::from_millis(1)) }
            return Err(aslt::hp::HpError::Unavailable("injected refusal".into()));
        }
        Ok(codes(l, sr))
    }));
    w.on_play(&a, SR);
    until(&mut w, &inside);
    w.on_play(&b, SR);
    release.store(true, Ordering::SeqCst);
    run(&mut w);
    assert_eq!(launched(&w), [("a.wav".into(), Hp), ("b.wav".into(), Hp)]);
    assert_eq!(shown(&w), Some(Hp));
    assert_eq!(w.analyzer.pre_cache_path().map(|p| p.to_path_buf()),
        Some(w.analyzer.effective_settings(&b).key(&w.analyzer.cache_dir()).1));
}

/// A retry is due, but the settings change before the worker's slot frees:
/// the retry lapses, and nothing is launched for settings no longer chosen.
#[test]
fn a_retry_lapses_when_the_settings_change_before_it_starts() {
    let d = Dir::new("lapse");
    let a = d.track("a.wav", 440.0);
    let mut w = window(&d.0.join("cache"), true);
    w.analyzer.hp.hook = Some(hook(|_, _, _, _, _| Err(aslt::hp::HpError::Unavailable("injected refusal".into()))));
    let gate = Gate::closed();
    w.analyzer.lifecycle_probe.exit_gate = Some(gate.clone());
    w.on_play(&a, SR);
    // Sent and consumed, the worker still alive: the retry is pending.
    let deadline = Instant::now() + Duration::from_secs(120);
    while !(gate.reached() && w.analyzer.pending_request.is_some()) {
        frame(&mut w);
        assert!(Instant::now() < deadline, "the retry was never queued");
        std::thread::sleep(Duration::from_millis(1));
    }
    w.analyzer.aslt_cfg.n_wavelets = 7;
    gate.open();
    run(&mut w);
    assert_eq!(launched(&w), [("a.wav".into(), Hp)], "a lapsed retry was launched");
}

// ---------------------------------------------------------------------------
// The real engine (GPU, run by hand)
// ---------------------------------------------------------------------------

/// The engine is made, used and closed on the analysis worker: after an
/// injected device failure the same worker has no current context, the
/// request is retried once under Standard, and HP ends for the session.
#[test]
#[ignore = "needs an NVIDIA GPU; run by hand"]
fn the_real_engine_is_torn_down_on_its_own_worker_after_a_device_failure() {
    let d = Dir::new("teardown");
    let a = d.track("a.wav", 440.0);
    let clean = Arc::new(Mutex::new(None));
    let seen = Arc::clone(&clean);
    let mut w = window(&d.0.join("cache"), true);
    w.analyzer.hp.hook = Some(hook(move |l, r, sr, abort, progress| {
        aslt::hp::NEXT_FAULTS.with(|f| f.set(Some(aslt::hp::Faults { launch: Some(5), ..Default::default() })));
        let got = aslt::hp::analyze_on_this_thread(l, r, sr, BARS, 20.0, 24_000.0, &aslt::hp::validated_cfg(), 180.0, abort, progress);
        *seen.lock().unwrap() = Some(aslt::hp::no_current_context());
        got
    }));
    w.on_play(&a, SR);
    run(&mut w);
    assert_eq!(*clean.lock().unwrap(), Some(Ok(true)), "the worker kept a context");
    assert_eq!(launched(&w), [("a.wav".into(), Hp), ("a.wav".into(), Standard)]);
    assert_eq!(shown(&w), Some(Standard));
    assert!(w.analyzer.hp_label().is_some_and(|l| l.contains("injected launch failure")), "{:?}", w.analyzer.hp_label());
}

/// One real round trip through the app: song1 (the owner's 48 kHz FLAC,
/// read only) analysed with High-accuracy GPU, published, played again from
/// the store, and every plane compared with the frozen native-route planes of
/// the same production decode (research round 8, `r8t_N`).
///
/// MOOSIK_HP_SONG1=<song1 .flac> MOOSIK_HP_REFS=<folder with G_mix/left/right.u16>
#[test]
#[ignore = "needs an NVIDIA GPU and the frozen references; run by hand"]
fn hp_app_round_trip_song1() {
    let song = PathBuf::from(std::env::var("MOOSIK_HP_SONG1").expect("MOOSIK_HP_SONG1"));
    let refs = PathBuf::from(std::env::var("MOOSIK_HP_REFS").expect("MOOSIK_HP_REFS"));
    // The rate the engine reports for the song's route (song1's by default).
    let rate: u32 = std::env::var("MOOSIK_HP_SONG1_RATE").ok().and_then(|v| v.parse().ok()).unwrap_or(48_000);
    let plane = |name: &str| -> Vec<u16> {
        let b = std::fs::read(refs.join(format!("G_{name}.u16"))).unwrap();
        b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    };
    let as_frames = |codes: Vec<u16>| {
        let frames = codes.len() / BARS;
        cache::PreFrames::from_v4_codes(codes, frames, BARS)
    };
    let (mix, left, right) = (as_frames(plane("mix")), as_frames(plane("left")), as_frames(plane("right")));
    let d = Dir::new("song1");
    let cache = d.0.join("cache");
    // MOOSIK_HP_COMPACT=1: the same round trip with the compact cache form
    // on. Run alone: the setting is process-wide.
    cache::set_compact(std::env::var("MOOSIK_HP_COMPACT").is_ok());
    let mut w = window(&cache, true);
    w.analyzer.pre_stereo_enabled = true;
    let t = Instant::now();
    w.on_play(&song, rate);
    run(&mut w);
    eprintln!("HP song1: analysed and published in {:.1}s", t.elapsed().as_secs_f64());
    assert_eq!(launched(&w), [(song.file_name().unwrap().to_string_lossy().into_owned(), Hp)]);
    assert_eq!(kinds(&w), ["done"]);
    assert_eq!(shown(&w), Some(Hp));
    assert!(*w.analyzer.pre_frames == mix, "Mix differs from the frozen native route");
    // The Mix view gives the channels back to the sidecar; a channel view
    // reads them from it, so Left and Right are checked as stored.
    let channels_as_stored = |w: &mut SpectrumWindow| {
        w.channel_view = channels::ChannelView::Split;
        let deadline = Instant::now() + Duration::from_secs(120);
        while !matches!(w.analyzer.pre_channels, PreChannels::Resident(_)) {
            frame(w);
            assert!(Instant::now() < deadline, "the channels were never read back");
            std::thread::sleep(Duration::from_millis(2));
        }
        let PreChannels::Resident(p) = &w.analyzer.pre_channels else { unreachable!() };
        assert!(*p.left() == left, "Left differs from the frozen native route");
        assert!(*p.right() == right, "Right differs from the frozen native route");
    };
    channels_as_stored(&mut w);
    assert!(w.analyzer.hp_label().is_some_and(|l| l.starts_with("High-accuracy GPU numerics")));

    // What the store holds for it.
    let mut sizes: Vec<(String, u64)> = std::fs::read_dir(cache.join("v2")).into_iter().flatten().flatten()
        .filter_map(|e| Some((e.path().extension()?.to_string_lossy().into_owned(), e.metadata().ok()?.len())))
        .filter(|(ext, _)| ext == "mix" || ext == "lr").collect();
    sizes.sort();
    eprintln!("HP song1: stored {sizes:?} (compact {})", cache::compact());

    // Read back from the store in a new window: the exact hit, nothing launched.
    let mut again = window(&cache, true);
    again.analyzer.pre_stereo_enabled = true;
    let t = Instant::now();
    again.on_play(&song, rate);
    eprintln!("HP song1: the exact hit's Mix was on screen {:.0} ms after the track change", t.elapsed().as_secs_f64() * 1e3);
    run(&mut again);
    assert!(launched(&again).is_empty());
    assert_eq!(shown(&again), Some(Hp));
    assert!(*again.analyzer.pre_frames == mix, "the stored Mix differs from the frozen native route");
    let t = Instant::now();
    channels_as_stored(&mut again);
    eprintln!("HP song1: Left and Right were read back {:.0} ms after asking", t.elapsed().as_secs_f64() * 1e3);
    cache::set_compact(false);
}

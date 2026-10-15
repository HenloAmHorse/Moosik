//! DSD decimation: the preprocessing feed and the playback fallback are two
//! settings, and the live tap's rate is what playback feeds it.
//!
//! Before 1.5.3 one setting drove the fallback decimation the listener hears
//! *and* the analysis feed, and the window took the live tap to run at the
//! analysis rate. These tests hold the separation from the spectrum side:
//! what a profile restores, what rate each consumer sees, and that a rate
//! change while an analysis runs behaves like any other settings change. The
//! engine side (`dsd_fallback_target`, the fallback source, the live rate the
//! app hands the window) is held in `main.rs`'s tests.
//!
//! Isolation as in `lifecycle_tests`: no window here reads the owner's
//! `spectrum.json` except through `new_isolated`, whose stored settings every
//! test overwrites before use; every analyser has a private cache directory;
//! inputs are generated DSF files in a temp folder.

use super::*;
use crate::dsd::decimate::{FALLBACK_RATES, LEGACY_ANALYSIS_RATE, PREPROCESS_RATES};
use std::time::{Duration, Instant};

const BARS: usize = 32;
const AT: f64 = 0.25;
const DSD64: u32 = 2_822_400;

struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "moosik_dsdrate_{tag}_{}_{:x}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0),
        ));
        std::fs::create_dir_all(&root).unwrap();
        Dir(root)
    }
}

impl Drop for Dir {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

/// A stereo DSF at `rate`: first-order sigma-delta of a tone per channel,
/// LSB-first as DSF stores it, whole 4096-byte blocks.
pub(crate) fn write_dsf(path: &Path, rate: u32, secs: f64, f_l: f64, f_r: f64) {
    let bs = 4096usize;
    let bytes = ((rate as f64 * secs / 8.0) as usize).div_ceil(bs) * bs;
    let modulate = |f: f64| -> Vec<u8> {
        let (mut acc, mut y) = (0.0f64, 0.0f64);
        (0..bytes).map(|i| {
            let mut b = 0u8;
            for k in 0..8 {
                let t = (i * 8 + k) as f64 / rate as f64;
                acc += 0.4 * (std::f64::consts::TAU * f * t).sin() - y;
                y = if acc >= 0.0 { 1.0 } else { -1.0 };
                b |= ((y > 0.0) as u8) << k;
            }
            b
        }).collect()
    };
    let (l, r) = (modulate(f_l), modulate(f_r));
    let rows: Vec<Vec<Vec<u8>>> = (0..bytes / bs)
        .map(|k| vec![l[k * bs..(k + 1) * bs].to_vec(), r[k * bs..(k + 1) * bs].to_vec()])
        .collect();
    let f = crate::dsd::tests::make_dsf(2, rate, 1, (bytes * 8) as u64, bs as u32, &rows, None);
    std::fs::write(path, f).unwrap();
}

/// A window in pre-process mode, Superlet, on a private cache directory, with
/// the DSD rates stated rather than inherited.
fn window(cache: &Path, preprocess: u32, fallback: u32) -> SpectrumWindow {
    let mut w = SpectrumWindow::new_isolated();
    w.mode = SpectrumMode::PreProcess;
    w.analyzer.cache_dir_override = Some(cache.to_path_buf());
    w.analyzer.cache_budget_gb = 0.0;
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
    let mut c = aslt::AsltPreset::Standard.config();
    c.q_ratio = 1.0;
    w.analyzer.aslt_cfg = c;
    w.analyzer.dsd_rate = preprocess;
    w.dsd_fallback_rate = fallback;
    w
}

fn frame(w: &mut SpectrumWindow, playing: bool) {
    w.last_fft_time = Some(Instant::now() - Duration::from_millis(20));
    w.tick(AT, playing);
}

/// Frames until nothing is running or waiting; counts frames on which a
/// picture other than nothing or `newest` was shown.
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

fn park(w: &mut SpectrumWindow) -> Arc<AtomicBool> {
    let hold = Arc::new(AtomicBool::new(true));
    w.analyzer.analysis_hold = Some(Arc::clone(&hold));
    hold
}

fn release(w: &mut SpectrumWindow, hold: &AtomicBool) {
    hold.store(false, Ordering::SeqCst);
    w.analyzer.analysis_hold = None;
}

/// A fresh analysis of `path` at `preprocess`, in its own directory.
fn oracle(d: &Dir, path: &Path, preprocess: u32) -> cache::PreFrames {
    let mut w = window(&d.0.join(format!("oracle_{preprocess}")), preprocess, FALLBACK_RATES[0]);
    w.on_play_live(path, DSD64 / 16, None);
    let (ok, _) = run(&mut w, false, &cache::PreFrames::default());
    assert!(ok, "the oracle never finished");
    let f = (*w.analyzer.pre_frames).clone();
    assert!(!f.is_empty(), "the oracle produced nothing");
    f
}

// ── What a profile restores ───────────────────────────────────────────────

/// Every kind of profile comes back with the rates it had, and only a profile
/// with no settings file at all starts at the new preprocessing default.
///
/// The playback fallback is never moved by the preprocessing setting: a file
/// from before the split restores its one rate to *both*, so neither what an
/// existing user hears nor what they see changes on update.
#[test]
fn a_profile_keeps_its_dsd_rates_through_the_split() {
    // Applying a stored profile pushes its GPU mode and worker count to the
    // calibration's globals; the fixture owns them and puts them back.
    let _calib = crate::spectrum::gpu_calib::isolated_test_env("dsd_rates_split");
    use crate::dsd::decimate::{DEFAULT_FALLBACK_RATE, DEFAULT_PREPROCESS_RATE};
    let rates = |w: &SpectrumWindow| (w.dsd_fallback_rate, w.analyzer.dsd_rate);

    // No file: playback keeps its old default, analysis takes the new one.
    let fresh = SpectrumWindow::new_fresh_profile();
    assert_eq!(rates(&fresh), (DEFAULT_FALLBACK_RATE, DEFAULT_PREPROCESS_RATE), "fresh profile");
    assert_eq!(DEFAULT_FALLBACK_RATE, LEGACY_ANALYSIS_RATE, "the audio default did not move");

    // Written before the split.
    for (json, want, what) in [
        ("{}", (LEGACY_ANALYSIS_RATE, LEGACY_ANALYSIS_RATE), "no rate field"),
        (r#"{"dsd_rate":176400}"#, (176_400, 176_400), "explicit 176.4"),
        (r#"{"dsd_rate":352800}"#, (352_800, 352_800), "explicit 352.8"),
        (r#"{"dsd_rate":12345}"#, (LEGACY_ANALYSIS_RATE, LEGACY_ANALYSIS_RATE), "unknown rate"),
    ] {
        let w = SpectrumWindow::new_from_settings_json(json);
        assert_eq!(rates(&w), want, "legacy file, {what}");
        // Loading is not a change: nothing is written back until the user
        // changes something.
        assert!(w.snapshot() == w.saved_settings, "legacy file, {what}: loading dirtied the profile");
    }

    // Written after it: the two are independent.
    for &fb in &FALLBACK_RATES {
        for &pre in &PREPROCESS_RATES {
            let json = format!(r#"{{"dsd_rate":{fb},"dsd_preprocess_rate":{pre}}}"#);
            let w = SpectrumWindow::new_from_settings_json(&json);
            assert_eq!(rates(&w), (fb, pre), "split file {json}");
            // And a save of it restores the same pair.
            let text = serde_json::to_string(&w.snapshot()).unwrap();
            assert_eq!(rates(&SpectrumWindow::new_from_settings_json(&text)), (fb, pre), "round trip {text}");
        }
    }
    // An unknown preprocessing value falls back to the file's own rate, as a
    // file without the field does — never to the fresh-profile default.
    let w = SpectrumWindow::new_from_settings_json(r#"{"dsd_rate":352800,"dsd_preprocess_rate":7}"#);
    assert_eq!(rates(&w), (352_800, 352_800));

    // Changing the preprocessing rate does not touch the playback rate.
    let mut w = SpectrumWindow::new_from_settings_json(r#"{"dsd_rate":176400}"#);
    w.set_dsd_preprocess_rate(352_800);
    assert_eq!(rates(&w), (176_400, 352_800));
    let text = serde_json::to_string(&w.snapshot()).unwrap();
    assert_eq!(rates(&SpectrumWindow::new_from_settings_json(&text)), (176_400, 352_800));
}

// ── Which rate each consumer sees ─────────────────────────────────────────

/// The analysis rate follows the preprocessing setting; the live rate follows
/// what playback feeds the tap, and a preprocessing change never moves it.
#[test]
fn the_preprocessing_rate_and_the_live_tap_rate_are_separate() {
    let d = Dir::new("separate");
    let dsf = d.0.join("t.dsf");
    write_dsf(&dsf, DSD64, 0.1, 1_000.0, 3_000.0);
    let mut w = window(&d.0.join("cache"), 352_800, 176_400);
    w.mode = SpectrumMode::RealTime;

    // Fallback playback at 176.4 feeds the tap; analysis at 352.8.
    w.on_play_live(&dsf, 176_400, Some(176_400));
    assert_eq!((w.analyzer.sample_rate, w.analyzer.live_rate), (352_800, 176_400));
    w.analyzer.abort_preprocess();

    for pre in [176_400, 352_800, 176_400] {
        w.set_dsd_preprocess_rate(pre);
        assert_eq!(w.analyzer.sample_rate, pre, "the analysis rate follows the setting");
        assert_eq!(w.analyzer.live_rate, 176_400, "a preprocessing change moved the live rate");
        assert_eq!(w.dsd_fallback_rate, 176_400, "a preprocessing change moved playback");
    }

    // No tap (DoP, native): nothing reads the live rate; it stands at the
    // analysis rate rather than at a carrier or byte rate.
    w.on_play_live(&dsf, DSD64 / 16, None);
    assert_eq!(w.analyzer.live_rate, w.analyzer.sample_rate);
    w.analyzer.abort_preprocess();

    // PCM: one rate for both, through the plain entry point too.
    let wav = d.0.join("t.wav");
    std::fs::write(&wav, b"").unwrap();
    w.on_play(&wav, 96_000);
    assert_eq!((w.analyzer.sample_rate, w.analyzer.live_rate), (96_000, 96_000));
    w.analyzer.abort_preprocess();
}

/// The real-time transform maps bins to Hz with the tap's rate.
///
/// A DSD window whose analysis feed is 352.8 kHz while playback feeds the tap
/// at 176.4 kHz must draw exactly what a 176.4 kHz PCM window draws from the
/// same samples — and not what a window mapping at 352.8 kHz draws.
#[test]
fn the_live_transform_maps_with_the_tap_rate() {
    let d = Dir::new("livemap");
    let dsf = d.0.join("t.dsf");
    write_dsf(&dsf, DSD64, 0.1, 1_000.0, 3_000.0);
    let pcm = d.0.join("t.wav");
    std::fs::write(&pcm, b"").unwrap();
    let samples: Vec<f32> = (0..65_536)
        .map(|i| (std::f32::consts::TAU * 5_000.0 * i as f32 / 176_400.0).sin() * 0.5)
        .collect();
    let bars = |path: &Path, sr: u32, live: Option<u32>| -> Vec<u32> {
        let mut w = window(&d.0.join("cache"), 352_800, 176_400);
        w.mode = SpectrumMode::RealTime;
        w.bar_mapping = BarMappingMode::Cqt;
        w.analyzer.bar_mapping = BarMappingMode::Cqt;
        w.on_play_live(path, sr, live);
        w.analyzer.abort_preprocess();
        // The same transform size whatever the rates, so only the mapping can
        // differ.
        w.analyzer.fft_size = 16_384;
        w.analyzer.rebuild_fft();
        w.sample_buf.lock().unwrap().extend_from_slice(&samples);
        w.analyzer.process_realtime(1.0);
        w.analyzer.magnitudes.iter().map(|m| m.to_bits()).collect()
    };
    let dsd_on_fallback = bars(&dsf, 176_400, Some(176_400));
    let pcm_176 = bars(&pcm, 176_400, Some(176_400));
    let pcm_352 = bars(&pcm, 352_800, Some(352_800));
    assert!(!dsd_on_fallback.is_empty());
    assert_eq!(dsd_on_fallback, pcm_176, "the live bars did not map at the tap's rate");
    assert_ne!(dsd_on_fallback, pcm_352, "the check cannot tell the two rates apart");
}

// ── A rate change is a settings change ────────────────────────────────────

/// The preprocessing rate changes while the analysis at the old rate is
/// running. As for any setting: nothing starts by itself, the old rate's
/// result is never installed, its cache is published under the old rate's
/// identity and stays reachable at that rate, and Re-analyze runs the new one.
/// Each cache records the rate its samples were actually produced at.
#[test]
fn a_rate_change_while_busy_never_installs_the_old_rate() {
    for playing in [false, true] {
        let d = Dir::new("busy");
        let dsf = d.0.join("t.dsf");
        write_dsf(&dsf, DSD64, 0.3, 700.0, 2_500.0);
        let truth_new = oracle(&d, &dsf, 176_400);
        let truth_old = oracle(&d, &dsf, 352_800);
        assert!(truth_new != truth_old, "the two rates are indistinguishable here");

        let cache_dir = d.0.join("cache");
        let mut w = window(&cache_dir, 352_800, 176_400);
        let hold = park(&mut w);
        w.on_play_live(&dsf, 176_400, Some(176_400));
        let (_, old_cache) = w.analyzer.effective_settings(&dsf).key(&cache_dir);
        w.set_dsd_preprocess_rate(176_400);
        assert!(w.needs_reanalysis, "playing={playing}: no Re-analyze banner");
        assert_eq!(w.analyzer.live_rate, 176_400, "playing={playing}: the live rate moved");
        release(&mut w, &hold);
        let (ok, stale) = run(&mut w, playing, &truth_new);
        assert!(ok, "playing={playing}: never settled");
        assert_eq!(stale, 0, "playing={playing}: the old rate's picture was installed");
        assert!(w.analyzer.pre_frames.is_empty(), "playing={playing}: something was installed");
        let record = crate::spectrum::unit_record(&old_cache).expect("the old rate's cache was published");
        assert!(record.contains("dsd=352800") && record.contains(";sr=352800;"),
            "playing={playing}: the old cache is not under the old rate: {record}");

        assert!(w.reanalyze(), "playing={playing}: Re-analyze refused: {}", w.status_msg);
        let (ok, stale) = run(&mut w, playing, &truth_new);
        assert!(ok && stale == 0, "playing={playing}: Re-analyze did not settle cleanly");
        assert!(*w.analyzer.pre_frames == truth_new, "playing={playing}: the new rate is not shown");
        let (_, new_cache) = w.analyzer.effective_settings(&dsf).key(&cache_dir);
        assert_ne!(new_cache, old_cache);
        let record = crate::spectrum::unit_record(&new_cache).expect("the new rate's cache");
        assert!(record.contains("dsd=176400") && record.contains(";sr=176400;"), "{record}");

        // Back to the old rate: its cache is an exact hit, not a re-analysis.
        w.set_dsd_preprocess_rate(352_800);
        assert!(!w.needs_reanalysis, "playing={playing}: the old rate's cache was not found");
        assert!(*w.analyzer.pre_frames == truth_old, "playing={playing}: the old rate's cache is not shown");
        assert_eq!(w.analyzer.live_rate, 176_400);
    }
}

// ── The 88.2 / 96 kHz preprocessing feed ──────────────────────────────────

/// A fresh profile analyses DSD at 88.2 kHz while its playback fallback stays
/// at 176.4 kHz. A 48k-family source requested at 88.2 kHz is analysed at the
/// 96 kHz the decimation produces, and everything about the analysis — its
/// record, hop and frame timing — is that produced rate's, not the label's.
/// Superlet keeps its configured frame rate at every feed rate.
#[test]
fn a_fresh_profile_analyses_dsd_at_88_2_and_a_48k_source_at_96() {
    let fresh = SpectrumWindow::new_fresh_profile();
    assert_eq!((fresh.dsd_fallback_rate, fresh.analyzer.dsd_rate), (176_400, 88_200));

    assert_eq!(dsd_rate_choice_label(88_200, Some(96_000)), "88.2 kHz (→ 96.0 kHz)");
    assert_eq!(dsd_rate_choice_label(88_200, Some(88_200)), "88.2 kHz");
    assert_eq!(dsd_rate_choice_label(88_200, None), "88.2 kHz");

    let d = Dir::new("fresh");
    for (dsd, produced) in [(DSD64, 88_200u32), (3_072_000, 96_000)] {
        let dsf = d.0.join(format!("t{dsd}.dsf"));
        write_dsf(&dsf, dsd, 0.2, 700.0, 2_500.0);
        let cache_dir = d.0.join(format!("cache{dsd}"));
        let mut w = window(&cache_dir, 88_200, 176_400);
        w.on_play_live(&dsf, dsd / 16, None);
        assert_eq!(w.analyzer.sample_rate, produced, "{dsd}: the analysis rate");
        let (ok, _) = run(&mut w, false, &cache::PreFrames::default());
        assert!(ok, "{dsd}: never settled");
        assert!(!w.analyzer.pre_frames.is_empty(), "{dsd}: nothing installed");
        let (_, file) = w.analyzer.effective_settings(&dsf).key(&cache_dir);
        let record = crate::spectrum::unit_record(&file).expect("published");
        let hop = aslt::hop_for_fps(produced, w.analyzer.pre_fps);
        assert!(record.contains("dsd=88200") && record.contains(&format!(";sr={produced};hop={hop}")),
            "{dsd}: {record}");
        assert_eq!(w.analyzer.pre_hop(), hop);
        let fps = produced as f64 / hop as f64;
        assert!((fps - w.analyzer.pre_fps as f64).abs() < 0.5,
            "{dsd}: the frame rate moved with the feed rate: {fps}");
    }
}

/// A cache analysed at a higher rate is never taken for the 88.2 kHz analysis:
/// not as an exact hit and not as an alternate. It stays on disk and is an
/// exact hit again at its own rate, and the 88.2 kHz analysis has its own.
#[test]
fn an_old_high_rate_cache_is_never_a_low_rate_hit() {
    let d = Dir::new("highrate");
    let dsf = d.0.join("t.dsf");
    write_dsf(&dsf, DSD64, 0.3, 700.0, 2_500.0);
    let truth_low = oracle(&d, &dsf, 88_200);
    let cache_dir = d.0.join("cache");
    let mut w = window(&cache_dir, 176_400, 176_400);
    w.on_play_live(&dsf, 176_400, Some(176_400));
    let (ok, _) = run(&mut w, false, &cache::PreFrames::default());
    assert!(ok);
    let high = (*w.analyzer.pre_frames).clone();
    assert!(high != truth_low, "the two rates are indistinguishable here");
    let (_, high_file) = w.analyzer.effective_settings(&dsf).key(&cache_dir);

    w.set_dsd_preprocess_rate(88_200);
    assert!(w.needs_reanalysis, "a high-rate cache was taken for the low rate");
    assert!(w.analyzer.pre_frames.is_empty(), "the high-rate picture is still shown");
    w.refresh_alternates();
    assert!(w.alt_notice.is_none(), "a high-rate cache was offered as an alternate");
    assert!(high_file.is_file(), "the high-rate cache was removed");

    assert!(w.reanalyze(), "Re-analyze refused: {}", w.status_msg);
    let (ok, stale) = run(&mut w, false, &truth_low);
    assert!(ok && stale == 0);
    assert!(*w.analyzer.pre_frames == truth_low, "the low rate's own analysis is not shown");

    w.set_dsd_preprocess_rate(176_400);
    assert!(!w.needs_reanalysis && *w.analyzer.pre_frames == high, "the high-rate cache is lost");
}

/// Channel views at 88.2 kHz, and across a rate change. With channels
/// produced, every view finds them at the rate they were analysed at; after a
/// change to a rate with no analysis there are none to show (never the other
/// rate's); back at 88.2 kHz they are found again. Paused and playing.
#[test]
fn channel_views_follow_the_preprocessing_rate() {
    use channels::ChannelView;
    for playing in [false, true] {
        let d = Dir::new("views");
        let dsf = d.0.join("t.dsf");
        write_dsf(&dsf, DSD64, 0.3, 700.0, 2_500.0);
        let mut w = window(&d.0.join("cache"), 88_200, 176_400);
        w.analyzer.pre_stereo_enabled = true;
        // A style that draws channels, whatever the profile had.
        w.style = VizStyle::Bars;
        w.on_play_live(&dsf, 176_400, Some(176_400));
        let (ok, _) = run(&mut w, playing, &cache::PreFrames::default());
        assert!(ok, "playing={playing}: never settled");
        assert!(w.analyzer.has_pre_channels(), "playing={playing}: no channels produced");

        let resident = |w: &mut SpectrumWindow, view: ChannelView| -> bool {
            w.channel_view = view;
            let deadline = Instant::now() + Duration::from_secs(60);
            while !w.analyzer.pre_channels_resident() && Instant::now() < deadline {
                frame(w, playing);
                std::thread::sleep(Duration::from_millis(2));
            }
            w.analyzer.pre_channels_resident()
        };
        for view in [ChannelView::Left, ChannelView::Right, ChannelView::Split, ChannelView::Diff] {
            assert!(resident(&mut w, view), "playing={playing}: {view:?} found no channels");
        }
        let pair = w.analyzer.pre_stereo().expect("resident");
        assert!(pair.left() != pair.right(), "playing={playing}: the channels are not distinct");
        w.channel_view = ChannelView::Mix;
        frame(&mut w, playing);

        w.set_dsd_preprocess_rate(176_400);
        frame(&mut w, playing);
        assert!(!w.analyzer.has_pre_channels() && !w.analyzer.pre_channels_resident(),
            "playing={playing}: channels shown at a rate that has no analysis");

        w.set_dsd_preprocess_rate(88_200);
        assert!(w.analyzer.has_pre_channels(), "playing={playing}: the 88.2 kHz channels were lost");
        assert!(resident(&mut w, ChannelView::Diff), "playing={playing}: Diff found no channels again");
    }
}

/// A track change while the 88.2 kHz analysis of a DSD64 track runs: the next
/// track, a 48k-family DSD file, is analysed at 96 kHz, and nothing of the
/// first is ever installed.
#[test]
fn a_track_change_mid_analysis_moves_to_the_next_tracks_rate() {
    for playing in [false, true] {
        let d = Dir::new("trackchange");
        let a = d.0.join("a.dsf");
        let b = d.0.join("b.dsf");
        write_dsf(&a, DSD64, 0.3, 700.0, 2_500.0);
        write_dsf(&b, 3_072_000, 0.3, 900.0, 3_100.0);
        let truth_b = oracle(&d, &b, 88_200);
        let mut w = window(&d.0.join("cache"), 88_200, 176_400);
        let hold = park(&mut w);
        w.on_play_live(&a, 176_400, Some(176_400));
        assert_eq!(w.analyzer.sample_rate, 88_200);
        w.on_play_live(&b, 192_000, Some(192_000));
        assert_eq!((w.analyzer.sample_rate, w.analyzer.live_rate), (96_000, 192_000));
        release(&mut w, &hold);
        let (ok, stale) = run(&mut w, playing, &truth_b);
        assert!(ok, "playing={playing}: never settled");
        assert_eq!(stale, 0, "playing={playing}: the first track's picture was installed");
        assert!(*w.analyzer.pre_frames == truth_b, "playing={playing}: the next track is not shown");
    }
}

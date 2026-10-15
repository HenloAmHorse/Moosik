//! The live FFT and the pre-process FFT are sized for their own feeds.
//!
//! Before round 6 one automatic size served both, chosen from the analysis
//! (pre-process) rate: on a DSD track analysed at 88.2 kHz while the PCM
//! fallback fed the tap at 176.4 kHz, the live transform was half as long as
//! intended, and changing the analysis-only DSD rate changed it. These tests
//! hold the separation (`SpectrumWindow::sync_fft_sizes`), the
//! consumer-specific duration labels (`fft_panel`), and the prospective cache
//! lookup behind the DSD rate buttons (`dsd_rate_cache_exists`), which must
//! look up exactly what the click will use.
//!
//! Isolation as in `dsd_rate_tests`: windows from `new_isolated` with every
//! field a test depends on set explicitly, private temp cache directories,
//! generated DSF inputs. No test forces an FFT size it means to observe.

use super::*;
use super::dsd_rate_tests::write_dsf;
use std::time::{Duration, Instant};

const DSD64: u32 = 2_822_400;
const DSD48: u32 = 3_072_000;

struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "moosik_fftdomain_{tag}_{}_{:x}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0),
        ));
        std::fs::create_dir_all(&root).unwrap();
        Dir(root)
    }
}

impl Drop for Dir {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

/// A window on a private cache directory with the given mapping and DSD
/// rates. Real-time mode unless a test analyses.
fn window(cache: &Path, mapping: BarMappingMode, preprocess: u32) -> SpectrumWindow {
    let mut w = SpectrumWindow::new_isolated();
    w.mode = SpectrumMode::RealTime;
    w.style = VizStyle::Bars;
    w.analyzer.cache_dir_override = Some(cache.to_path_buf());
    w.analyzer.cache_budget_gb = 0.0;
    w.analyzer.pre_stereo_enabled = false;
    w.bar_count = 32;
    w.analyzer.bar_count = 32;
    w.bar_mapping = mapping.clone();
    w.analyzer.bar_mapping = mapping;
    w.pad_factor = 2;
    w.analyzer.pad_factor = 2;
    w.overlap = 0.5;
    w.analyzer.overlap = 0.5;
    w.window_fn = WindowFn::Hann;
    w.analyzer.window_fn = WindowFn::Hann;
    w.min_freq = 200.0;
    w.max_freq = 16_000.0;
    w.analyzer.min_freq = 200.0;
    w.analyzer.max_freq = 16_000.0;
    w.analyzer.aslt_preset = None;
    let mut c = aslt::AsltPreset::Standard.config();
    c.q_ratio = 1.0;
    w.analyzer.aslt_cfg = c;
    w.analyzer.dsd_rate = preprocess;
    w.dsd_fallback_rate = 176_400;
    w
}

fn run_to_idle(w: &mut SpectrumWindow) {
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut quiet = 0;
    while quiet < 3 {
        w.last_fft_time = Some(Instant::now() - Duration::from_millis(20));
        w.tick(0.1, false);
        quiet = if w.analyzer.analysis_idle() { quiet + 1 } else { 0 };
        assert!(Instant::now() < deadline, "never settled");
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn refresh_cache_set(w: &mut SpectrumWindow) {
    w.cache_file_set = cached_units(&w.analyzer.cache_dir());
}

/// The live bars a window draws from `samples` through its own live
/// transform, at whatever size its policy chose.
fn live_bars(w: &mut SpectrumWindow, samples: &[f32]) -> Vec<u32> {
    {
        let mut b = w.sample_buf.lock().unwrap();
        b.clear();
        b.extend_from_slice(samples);
    }
    w.analyzer.process_realtime(1.0);
    w.analyzer.magnitudes.iter().map(|m| m.to_bits()).collect()
}

fn tone(rate: u32, hz: f32, n: usize) -> Vec<f32> {
    (0..n).map(|i| (std::f32::consts::TAU * hz * i as f32 / rate as f32).sin() * 0.5).collect()
}

// ── Automatic sizing per consumer ─────────────────────────────────────────

/// Automatic sizes follow each consumer's own feed, for both families: the
/// pre-process at the produced analysis rate, the live transform at the tap.
/// And the live bars are exactly a PCM window's at the tap's rate — sizes
/// chosen by the policy, not forced.
#[test]
fn automatic_sizes_follow_each_consumers_feed() {
    let d = Dir::new("auto");
    for (dsd, tap, pre_rate) in [(DSD64, 176_400u32, 88_200u32), (DSD48, 192_000, 96_000)] {
        let dsf = d.0.join(format!("t{dsd}.dsf"));
        write_dsf(&dsf, dsd, 0.05, 1_000.0, 2_000.0);
        let mut w = window(&d.0.join("c"), BarMappingMode::Cqt, 88_200);
        w.on_play_live(&dsf, tap, Some(tap));
        w.analyzer.abort_preprocess();
        assert_eq!(w.analyzer.sample_rate, pre_rate);
        assert_eq!(w.analyzer.pre_fft_size, auto_fft_size_for(pre_rate), "{dsd}: pre-process size");
        assert_eq!(w.analyzer.pre_fft_size, 16_384);
        assert_eq!(w.analyzer.fft_size, auto_fft_size_for(tap), "{dsd}: live size");
        assert_eq!(w.analyzer.fft_size, 32_768);

        // The same samples through a PCM window whose single rate is the tap's.
        let wav = d.0.join("t.wav");
        std::fs::write(&wav, b"").unwrap();
        let mut pcm = window(&d.0.join("c"), BarMappingMode::Cqt, 88_200);
        pcm.on_play(&wav, tap);
        pcm.analyzer.abort_preprocess();
        let sig = tone(tap, 9_000.0, 65_536);
        assert_eq!(live_bars(&mut w, &sig), live_bars(&mut pcm, &sig),
                   "{dsd}: the live transform is not the tap-rate one");
    }
}

/// The pre-process rate moves 88.2 → 176.4 → 352.8 kHz with the tap fixed at
/// 176.4 kHz: the pre-process size follows it, the live size, plan and drawn
/// bars do not move.
#[test]
fn a_preprocessing_rate_change_leaves_the_live_transform_alone() {
    let d = Dir::new("prechange");
    let dsf = d.0.join("t.dsf");
    write_dsf(&dsf, DSD64, 0.05, 1_000.0, 2_000.0);
    let mut w = window(&d.0.join("c"), BarMappingMode::Cqt, 88_200);
    w.on_play_live(&dsf, 176_400, Some(176_400));
    w.analyzer.abort_preprocess();
    let sig = tone(176_400, 9_000.0, 65_536);
    let before = live_bars(&mut w, &sig);
    let live_size = w.analyzer.fft_size;
    for (rate, pre) in [(176_400u32, 32_768usize), (352_800, 32_768), (88_200, 16_384)] {
        w.set_dsd_preprocess_rate(rate);
        assert_eq!(w.analyzer.pre_fft_size, pre, "pre-process size at {rate}");
        assert_eq!(w.fft_size, pre);
        assert_eq!(w.analyzer.fft_size, live_size, "live size moved at {rate}");
        assert_eq!(w.analyzer.live_rate, 176_400);
        assert_eq!(live_bars(&mut w, &sig), before, "live bars moved at {rate}");
    }
}

/// A different live route at track start with the same descriptor: the
/// pre-process size, identity and cache key do not depend on the tap.
#[test]
fn the_live_route_does_not_reach_the_preprocessing_identity() {
    let d = Dir::new("liveroute");
    let dsf = d.0.join("t.dsf");
    write_dsf(&dsf, DSD64, 0.05, 1_000.0, 2_000.0);
    let mut keys = Vec::new();
    for live in [None, Some(176_400u32), Some(352_800)] {
        let mut w = window(&d.0.join("c"), BarMappingMode::Cqt, 88_200);
        w.on_play_live(&dsf, 176_400, live);
        w.analyzer.abort_preprocess();
        keys.push((w.analyzer.pre_fft_size, w.analyzer.effective_settings(&dsf).key(&d.0).1));
        assert_eq!(w.analyzer.fft_size, auto_fft_size_for(live.unwrap_or(88_200)));
    }
    assert!(keys.windows(2).all(|k| k[0] == k[1]), "the tap moved the pre-process identity: {keys:?}");
}

/// A size chosen by hand is one size for both consumers, and no pre-process
/// rate change overwrites it. A track start returns to automatic, as it
/// always has.
#[test]
fn a_manual_size_is_kept_across_rate_changes() {
    let d = Dir::new("manual");
    let dsf = d.0.join("t.dsf");
    write_dsf(&dsf, DSD64, 0.05, 1_000.0, 2_000.0);
    let mut w = window(&d.0.join("c"), BarMappingMode::Cqt, 88_200);
    w.on_play_live(&dsf, 176_400, Some(176_400));
    w.analyzer.abort_preprocess();
    // What the panel's click does.
    w.fft_size = 4_096;
    w.auto_fft_size = 0;
    w.sync_fft_sizes();
    assert_eq!((w.analyzer.pre_fft_size, w.analyzer.fft_size), (4_096, 4_096));
    for rate in [176_400u32, 352_800, 88_200] {
        w.set_dsd_preprocess_rate(rate);
        assert_eq!((w.fft_size, w.analyzer.pre_fft_size, w.analyzer.fft_size), (4_096, 4_096, 4_096),
                   "a rate change overwrote the manual size at {rate}");
        assert_eq!(w.auto_fft_size, 0, "manual became automatic at {rate}");
    }
    // Unchanged policy: the next track start is automatic again.
    w.on_play_live(&dsf, 176_400, Some(176_400));
    w.analyzer.abort_preprocess();
    assert_ne!(w.auto_fft_size, 0);
    assert_eq!((w.analyzer.pre_fft_size, w.analyzer.fft_size), (16_384, 32_768));
}

/// Each panel quotes its own consumer's duration.
#[test]
fn fft_durations_are_quoted_for_the_feed_they_describe() {
    assert!((fft_window_ms(16_384, 176_400) - 92.88).abs() < 0.01);
    let d = Dir::new("labels");
    let dsf = d.0.join("t.dsf");
    write_dsf(&dsf, DSD64, 0.05, 1_000.0, 2_000.0);
    let mut w = window(&d.0.join("c"), BarMappingMode::Cqt, 88_200);
    w.on_play_live(&dsf, 176_400, Some(176_400));
    w.analyzer.abort_preprocess();
    w.fft_size = 16_384;
    w.auto_fft_size = 0;
    w.sync_fft_sizes();
    // Live panels: the Real-time view and the Superlet live-preview fold.
    w.mode = SpectrumMode::RealTime;
    let (sr, size, auto) = w.fft_panel(true);
    assert_eq!((sr, size, auto), (176_400, 16_384, 0));
    assert!((fft_window_ms(size, sr) - 92.88).abs() < 0.01, "live 16384 @ 176.4 kHz is ~92.9 ms");
    w.mode = SpectrumMode::PreProcess;
    assert_eq!(w.fft_panel(false).0, 176_400, "the Superlet live preview is live");
    // The FFT pre-process panel: the analysis feed.
    let (sr, size, _) = w.fft_panel(true);
    assert_eq!((sr, size), (88_200, 16_384));
    assert!((fft_window_ms(size, sr) - 185.76).abs() < 0.01);
    // Automatic: each panel stars its own consumer's size.
    w.on_play_live(&dsf, 176_400, Some(176_400));
    w.analyzer.abort_preprocess();
    w.mode = SpectrumMode::RealTime;
    assert_eq!(w.fft_panel(true).2, 32_768);
    w.mode = SpectrumMode::PreProcess;
    assert_eq!(w.fft_panel(true).2, 16_384);
    // 48 kHz family: produced 96 kHz analysis, 192 kHz tap.
    let d48 = d.0.join("t48.dsf");
    write_dsf(&d48, DSD48, 0.05, 1_000.0, 2_000.0);
    w.on_play_live(&d48, 192_000, Some(192_000));
    w.analyzer.abort_preprocess();
    assert_eq!(w.fft_panel(true).0, 96_000);
    w.mode = SpectrumMode::RealTime;
    assert_eq!(w.fft_panel(true).0, 192_000);
}

// ── The DSD rate buttons' cache indicator ─────────────────────────────────

/// FFT (constant-Q) pre-process: the indicator for a rate looks up the
/// analysis the click would use — that rate's produced feed and its
/// automatic size, or the manual size — and the click then reuses it.
#[test]
fn the_rate_indicator_finds_what_the_click_will_use() {
    let d = Dir::new("indicator");
    for (dsd, high_produced) in [(DSD64, 176_400u32), (DSD48, 192_000)] {
        // Long enough for a 32768-point frame at 88.2 kHz.
        let dsf = d.0.join(format!("t{dsd}.dsf"));
        write_dsf(&dsf, dsd, 0.6, 700.0, 2_500.0);
        let cache = d.0.join(format!("cache{dsd}"));

        // A cache at 176.4 (its automatic size, 32768).
        let mut w = window(&cache, BarMappingMode::Cqt, 176_400);
        w.mode = SpectrumMode::PreProcess;
        w.on_play_live(&dsf, 176_400, Some(176_400));
        run_to_idle(&mut w);
        assert!(!w.analyzer.pre_frames.is_empty(), "{dsd}: no analysis");
        assert_eq!(w.analyzer.sample_rate, high_produced);
        let high_frames = (*w.analyzer.pre_frames).clone();

        // Now at 88.2 (automatic 16384): nothing cached there.
        w.set_dsd_preprocess_rate(88_200);
        assert!(w.needs_reanalysis);
        refresh_cache_set(&mut w);
        // The old lookup used the current size (16384) and missed this.
        assert!(w.dsd_rate_cache_exists(176_400), "{dsd}: a valid 176.4 cache is not indicated");
        assert!(!w.dsd_rate_cache_exists(88_200));
        // And the click reuses exactly it.
        w.set_dsd_preprocess_rate(176_400);
        assert!(!w.needs_reanalysis, "{dsd}: the indicated cache was not used");
        assert!(*w.analyzer.pre_frames == high_frames);

        // A manual 32768 analysis at 88.2: a cache at the wrong size for an
        // automatic 88.2 choice.
        w.set_dsd_preprocess_rate(88_200);
        w.fft_size = 32_768;
        w.auto_fft_size = 0;
        w.sync_fft_sizes();
        w.try_load_or_flag_reanalysis();
        assert!(w.reanalyze(), "{dsd}: {}", w.status_msg);
        run_to_idle(&mut w);
        refresh_cache_set(&mut w);
        drop(w);

        // Automatic again (a track start): the manual-size 88.2 cache is the
        // wrong size for an automatic 88.2 choice and must not be advertised.
        let mut w = window(&cache, BarMappingMode::Cqt, 176_400);
        w.mode = SpectrumMode::PreProcess;
        w.on_play_live(&dsf, 176_400, Some(176_400));
        run_to_idle(&mut w);
        refresh_cache_set(&mut w);
        assert!(!w.dsd_rate_cache_exists(88_200),
                "{dsd}: a 32768-point 88.2 cache was advertised for an automatic (16384) choice");
        // While under the manual 32768 policy it is exactly the one.
        w.fft_size = 32_768;
        w.auto_fft_size = 0;
        w.sync_fft_sizes();
        assert!(w.dsd_rate_cache_exists(88_200), "{dsd}: the manual-size cache is not indicated");
        w.set_dsd_preprocess_rate(88_200);
        assert!(!w.needs_reanalysis, "{dsd}: the manual-size cache was not used");
        assert_eq!(w.auto_fft_size, 0, "{dsd}: manual became automatic");
    }
}

/// Superlet: the FFT fields are not part of its identity, so the indicator
/// and the reuse are what they were.
#[test]
fn superlet_rate_indicator_and_reuse_are_unchanged() {
    let d = Dir::new("superlet");
    let dsf = d.0.join("t.dsf");
    write_dsf(&dsf, DSD64, 0.3, 700.0, 2_500.0);
    let cache = d.0.join("cache");
    let mut w = window(&cache, BarMappingMode::Superlet, 176_400);
    w.mode = SpectrumMode::PreProcess;
    w.on_play_live(&dsf, 176_400, Some(176_400));
    run_to_idle(&mut w);
    let high = (*w.analyzer.pre_frames).clone();
    let (_, high_key) = w.analyzer.effective_settings(&dsf).key(&cache);
    w.set_dsd_preprocess_rate(88_200);
    refresh_cache_set(&mut w);
    assert!(w.dsd_rate_cache_exists(176_400));
    for (manual, size) in [(false, 0usize), (true, 4_096)] {
        if manual {
            w.fft_size = size;
            w.auto_fft_size = 0;
            w.sync_fft_sizes();
        }
        let prospective = w.prospective_dsd_settings(&dsf, 176_400).unwrap().key(&cache).1;
        assert_eq!(prospective, high_key, "manual={manual}: the superlet key depends on an FFT size");
        assert!(w.dsd_rate_cache_exists(176_400));
    }
    w.set_dsd_preprocess_rate(176_400);
    assert!(!w.needs_reanalysis && *w.analyzer.pre_frames == high);
}

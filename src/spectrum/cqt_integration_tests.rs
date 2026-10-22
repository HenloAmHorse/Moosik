//! Genuine CQT through profile, job, cache and presentation entry points.
//! Inputs and caches are private generated fixtures; no player/device is opened.

use super::*;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const SR: u32 = 48_000;
const SAMPLES: usize = 16_000;
const BARS: usize = 16;

struct Fixture {
    root: PathBuf,
    track: PathBuf,
    cache: PathBuf,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "moosik_true_cqt_{tag}_{}_{:x}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(&root).unwrap();
        let track = root.join("antiphase.wav");
        let cache = root.join("cache");
        // Exact opposite i16 values survive decoding as exact opposites too.
        let mut data = Vec::with_capacity(SAMPLES * 4);
        for i in 0..SAMPLES {
            let phase = std::f64::consts::TAU * 880.0 * i as f64 / SR as f64;
            let left = (phase.sin() * 12_000.0) as i16;
            data.extend_from_slice(&left.to_le_bytes());
            data.extend_from_slice(&(-left).to_le_bytes());
        }
        let mut wav = Vec::with_capacity(44 + data.len());
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&((36 + data.len()) as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&SR.to_le_bytes());
        wav.extend_from_slice(&(SR * 4).to_le_bytes());
        wav.extend_from_slice(&4u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);
        std::fs::write(&track, wav).unwrap();
        Self { root, track, cache }
    }

    fn heads(&self) -> usize {
        std::fs::read_dir(identity::store_dir(&self.cache))
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().ends_with(".head"))
            .count()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn config() -> cqt::Config {
    cqt::Config {
        bins_per_octave: 12,
        window: cqt::Window::Hann,
        frame_rate: 180.0,
        bass_limit_s: None,
    }
}

fn window(cache: &Path) -> SpectrumWindow {
    // Unlike new_isolated, this also avoids reading spectrum.json.
    let mut w = SpectrumWindow::new_fresh_profile();
    w.mode = SpectrumMode::PreProcess;
    w.bar_mapping = BarMappingMode::TrueCqt(config());
    w.last_true_cqt = config();
    w.bar_count = BARS;
    w.min_freq = 220.0;
    w.max_freq = 8_000.0;
    w.analyzer.bar_mapping = w.bar_mapping.clone();
    w.analyzer.bar_count = BARS;
    w.analyzer.min_freq = w.min_freq;
    w.analyzer.max_freq = w.max_freq;
    w.analyzer.sample_rate = SR;
    w.analyzer.cache_dir_override = Some(cache.to_path_buf());
    w.analyzer.cache_budget_gb = 0.0;
    w.analyzer.pre_stereo_enabled = true;
    w
}

fn settle(a: &mut SpectrumAnalyzer) {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        a.service_preprocess();
        if a.analysis_idle() {
            return;
        }
        assert!(Instant::now() < deadline, "CQT worker did not retire");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn spawned(a: &SpectrumAnalyzer) -> usize {
    a.lifecycle_probe.spawned.load(Ordering::SeqCst)
}

#[test]
fn old_cqt_profiles_keep_fft_weighting_and_its_frozen_identity() {
    let _calib = gpu_calib::isolated_test_env("true_cqt_old_profile");
    let old: SpectrumSettings = serde_json::from_str(r#"{"bar_mapping":"Cqt"}"#).unwrap();
    assert_eq!(old.bar_mapping, BarMappingMode::Cqt);
    assert!(old.last_true_cqt.is_none());
    assert_eq!(serde_json::to_string(&old.bar_mapping).unwrap(), "\"Cqt\"");
    let w = SpectrumWindow::new_from_settings_json(r#"{"bar_mapping":"Cqt"}"#);
    assert_eq!(w.bar_mapping, BarMappingMode::Cqt);

    let mut aslt = aslt::AsltPreset::Standard.config();
    aslt.scale = freq_scale::FreqScale::Log;
    let s = settings_for(
        Path::new("C:/music/old.flac"),
        1024,
        8192,
        2,
        0.5,
        &WindowFn::Hann,
        20.0,
        24_000.0,
        &BarMappingMode::Cqt,
        &InterpolationMode::None,
        176_400,
        &aslt,
        180.0,
    );
    let canonical = s.canonical();
    assert_eq!(
        canonical.split_once(";bars=").unwrap().1,
        "1024;lo=41a00000;hi=46bb8000;map=cqw;dsd=-;fft=8192;pad=2;ov=3f000000;window=hann;interp=none;scale=log",
        "adding genuine CQT changed the existing FFT-weighted Cqt identity",
    );
}

#[test]
fn cqt_profile_remembers_its_config_without_overwriting_superlet_scale() {
    let _calib = gpu_calib::isolated_test_env("true_cqt_profile");
    let mut w = SpectrumWindow::new_fresh_profile();
    let cfg = cqt::Config {
        bins_per_octave: 72,
        window: cqt::Window::BlackmanHarris,
        frame_rate: 137.5,
        bass_limit_s: None,
    };
    w.analyzer.aslt_preset = None;
    w.analyzer.aslt_cfg.scale = freq_scale::FreqScale::Erb;
    w.last_true_cqt = cfg;
    w.bar_mapping = BarMappingMode::TrueCqt(cfg);
    w.analyzer.bar_mapping = w.bar_mapping.clone();
    assert_eq!(w.analyzer.analysis_scale(), freq_scale::FreqScale::Log);
    assert_eq!(w.analyzer.aslt_cfg.scale, freq_scale::FreqScale::Erb);

    // An otherwise valid stored profile may have selected the unavailable mode.
    let mut active = w.snapshot();
    active.mode = SpectrumMode::RealTime;
    let active_json = serde_json::to_string(&active).unwrap();
    let active_back = SpectrumWindow::new_from_settings_json(&active_json);
    assert_eq!(active_back.mode, SpectrumMode::PreProcess);
    assert_eq!(active_back.bar_mapping, BarMappingMode::TrueCqt(cfg));
    assert_eq!(active_back.last_true_cqt, cfg);
    assert_eq!(
        active_back.analyzer.analysis_scale(),
        freq_scale::FreqScale::Log
    );
    assert_eq!(
        active_back.analyzer.aslt_cfg.scale,
        freq_scale::FreqScale::Erb
    );

    // Leaving CQT must not erase its last settings when the app is closed.
    w.bar_mapping = BarMappingMode::Cqt;
    w.analyzer.bar_mapping = BarMappingMode::Cqt;
    let inactive_json = serde_json::to_string(&w.snapshot()).unwrap();
    let inactive_back = SpectrumWindow::new_from_settings_json(&inactive_json);
    assert_eq!(inactive_back.bar_mapping, BarMappingMode::Cqt);
    assert_eq!(inactive_back.last_true_cqt, cfg);
    assert_eq!(
        inactive_back.analyzer.analysis_scale(),
        freq_scale::FreqScale::Erb
    );
}

struct HeadlessUi {
    ctx: egui::Context,
    time: f64,
}

impl HeadlessUi {
    fn new() -> Self {
        let ctx = egui::Context::default();
        ctx.set_embed_viewports(true);
        Self { ctx, time: 0.0 }
    }

    fn frame(&mut self, w: &mut SpectrumWindow, events: Vec<egui::Event>) -> Vec<(String, Rect)> {
        self.time += 1.0 / 60.0;
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(1800.0, 1600.0))),
            max_texture_side: Some(8192),
            time: Some(self.time),
            events,
            ..Default::default()
        };
        let output = self.ctx.run(input, |ctx| w.show(ctx));
        fn collect(shape: &egui::Shape, labels: &mut Vec<(String, Rect)>) {
            match shape {
                egui::Shape::Text(t) => labels.push((
                    t.galley.text().to_string(),
                    Rect::from_min_size(t.pos, t.galley.size()),
                )),
                egui::Shape::Vec(shapes) => {
                    for shape in shapes {
                        collect(shape, labels);
                    }
                }
                _ => {}
            }
        }
        let mut labels = Vec::new();
        for shape in output.shapes {
            collect(&shape.shape, &mut labels);
        }
        labels
    }

    fn click_label(&mut self, w: &mut SpectrumWindow, label: &str) -> Vec<(String, Rect)> {
        // A prior frame establishes the actual layout; no hard-coded hit points.
        let labels = self.frame(w, Vec::new());
        let pos = labels
            .iter()
            .find(|(text, _)| text == label)
            .unwrap_or_else(|| panic!("UI did not draw {label:?}"))
            .1
            .center();
        self.frame(
            w,
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ],
        );
        self.frame(
            w,
            vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        )
    }
}

#[test]
fn selecting_true_cqt_uses_the_remembered_config_and_hides_fft_controls() {
    let _calib = gpu_calib::isolated_test_env("true_cqt_selector");
    let f = Fixture::new("selector");
    let mut w = window(&f.cache);
    let cfg = cqt::Config {
        bins_per_octave: 72,
        frame_rate: 137.5,
        ..config()
    };
    w.bar_mapping = BarMappingMode::Cqt;
    w.analyzer.bar_mapping = BarMappingMode::Cqt;
    w.mode = SpectrumMode::RealTime;
    w.style = VizStyle::Spectrogram;
    w.analyzer.aslt_preset = None;
    w.analyzer.aslt_cfg.scale = freq_scale::FreqScale::Erb;
    w.last_true_cqt = cfg;
    w.open = true;
    w.show_fft_settings = true;
    let mut ui = HeadlessUi::new();
    ui.frame(&mut w, Vec::new());
    let labels = ui.click_label(&mut w, "True CQT");
    assert_eq!(w.bar_mapping, BarMappingMode::TrueCqt(cfg));
    assert_eq!(w.analyzer.bar_mapping, w.bar_mapping);
    assert_eq!(w.mode, SpectrumMode::PreProcess);
    assert_eq!(w.style, VizStyle::Waterfall);
    assert_eq!(w.analyzer.analysis_scale(), freq_scale::FreqScale::Log);
    assert_eq!(w.analyzer.aslt_cfg.scale, freq_scale::FreqScale::Erb);
    assert!(!w.fft_knobs_apply());
    for irrelevant in [
        "FFT size:",
        "Interpolation:",
        "Zero-padding:",
        "Overlap:",
    ] {
        assert!(
            !labels.iter().any(|(text, _)| text == irrelevant),
            "true CQT exposed an inapplicable control: {irrelevant}"
        );
    }
    assert!(labels.iter().any(|(text, _)| text == "Resolution:"));
    assert!(labels.iter().any(|(text, _)| text == "Live FFT preview"));

    ui.click_label(&mut w, "FFT · constant-Q weighting");
    assert_eq!(w.bar_mapping, BarMappingMode::Cqt);
    assert_eq!(w.last_true_cqt, cfg);
    assert_eq!(w.analyzer.analysis_scale(), freq_scale::FreqScale::Erb);
    assert!(w.fft_knobs_apply());
    ui.click_label(&mut w, "True CQT");
    assert_eq!(w.bar_mapping, BarMappingMode::TrueCqt(cfg));
    assert!(
        w.analyzer.analysis_idle(),
        "method selection without a track started work"
    );
}

#[test]
fn drawing_cqt_controls_preserves_valid_clocks_outside_the_slider_range() {
    let _calib = gpu_calib::isolated_test_env("true_cqt_slider_range");
    for frame_rate in [1.0, 360.0] {
        let cfg = cqt::Config {
            frame_rate,
            ..config()
        };
        let mut w = SpectrumWindow::new_fresh_profile();
        w.bar_mapping = BarMappingMode::TrueCqt(cfg);
        w.last_true_cqt = cfg;
        let ctx = egui::Context::default();
        let mut changed = false;
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                changed |= w.true_cqt_settings_ui(ui);
            });
        });
        assert!(!changed, "drawing controls silently changed a valid clock");
        assert_eq!(w.bar_mapping, BarMappingMode::TrueCqt(cfg));
        assert_eq!(w.last_true_cqt, cfg);
    }
}

#[test]
fn invalid_stored_cqt_configs_are_sanitized_before_identity_or_analysis() {
    let _calib = gpu_calib::isolated_test_env("true_cqt_bad_profile");
    let mut w = SpectrumWindow::new_fresh_profile();
    let invalid = cqt::Config {
        bins_per_octave: 0,
        frame_rate: 0.0,
        ..config()
    };
    w.bar_mapping = BarMappingMode::TrueCqt(invalid);
    w.last_true_cqt = invalid;
    let json = serde_json::to_string(&w.snapshot()).unwrap();
    let back = SpectrumWindow::new_from_settings_json(&json);
    assert_eq!(
        back.bar_mapping,
        BarMappingMode::TrueCqt(cqt::Config::default())
    );
    assert_eq!(back.last_true_cqt, cqt::Config::default());
    assert_eq!(back.mode, SpectrumMode::PreProcess);
    assert!(back.last_true_cqt.validate().is_ok());
}

#[test]
fn cqt_display_axes_and_loudness_use_the_clamped_log_grid_only_for_cqt() {
    let _calib = gpu_calib::isolated_test_env("true_cqt_axis");
    let mut w = SpectrumWindow::new_fresh_profile();
    w.analyzer.sample_rate = 44_100;
    w.max_freq = 24_000.0;
    w.min_freq = 20.0;
    w.loudness_mode = LoudnessMode::EqualLoudness;
    w.analyzer.aslt_cfg.scale = freq_scale::FreqScale::Erb;
    w.bar_mapping = BarMappingMode::TrueCqt(config());
    w.analyzer.bar_mapping = w.bar_mapping.clone();
    w.sync_params();
    assert_eq!(w.analysis_max_freq(), 22_050.0);
    assert_eq!(w.analysis_scale(), freq_scale::FreqScale::Log);
    assert_eq!(
        w.analyzer.eq_weights,
        compute_eq_weights(w.bar_count, 20.0, 22_050.0, freq_scale::FreqScale::Log)
    );
    // Merely selecting CQT did not replace the old routes' geometry settings.
    w.bar_mapping = BarMappingMode::Cqt;
    w.analyzer.bar_mapping = w.bar_mapping.clone();
    w.sync_params();
    assert_eq!(w.analysis_max_freq(), 24_000.0);
    assert_eq!(w.analysis_scale(), freq_scale::FreqScale::Erb);
}

fn live_tone(samples: usize, sample_rate: u32) -> Vec<f32> {
    (0..samples).map(|i| {
        0.4 * (std::f32::consts::TAU * 880.0 * i as f32 / sample_rate as f32).sin()
    }).collect()
}

fn preview_analyzer(mapping: BarMappingMode) -> SpectrumAnalyzer {
    let mut a = SpectrumAnalyzer::new(new_sample_buf());
    a.fft_size = 256;
    a.rebuild_fft();
    a.sample_rate = 44_100;
    a.live_rate = 96_000;
    a.bar_count = BARS;
    a.min_freq = 220.0;
    a.max_freq = 24_000.0;
    a.aslt_cfg.scale = freq_scale::FreqScale::Log;
    a.bar_mapping = mapping;
    *a.sample_buf.lock().unwrap() = live_tone(1024, a.live_rate);
    a
}

#[test]
fn pending_cqt_preview_is_the_existing_fft_weighting_and_reuses_its_plan() {
    let _calib = gpu_calib::isolated_test_env("true_cqt_fft_preview");
    let selected = BarMappingMode::TrueCqt(config());
    let mut preview = preview_analyzer(selected.clone());
    let mut fft = preview_analyzer(BarMappingMode::Cqt);
    for frame in 0..3 {
        preview.tick_pre(frame as f64 / 60.0, 1.0 / 60.0);
        fft.process_realtime(1.0 / 60.0);
        assert_eq!(preview.magnitudes, fft.magnitudes);
        assert_eq!(preview.last_fft_norms, fft.last_fft_norms);
        assert_eq!(preview.bar_mapping, selected);
        assert!(preview.pre_frames.is_empty());
        assert!(preview.pre_record().is_none());
        assert!(preview.pre_cache().is_none());
    }
    assert!(preview.magnitudes.iter().any(|&v| v > 0.0));
    let buffers = (preview.rt_samples.as_ptr(), preview.rt_work.as_ptr(),
        preview.rt_scratch.as_ptr(), preview.last_fft_norms.as_ptr());
    let plans = preview.cqt_plan_builds;
    for frame in 3..12 {
        preview.tick_pre(frame as f64 / 60.0, 1.0 / 60.0);
    }
    assert_eq!(preview.cqt_plan_builds, plans, "preview rebuilt its gather plan per tick");
    assert_eq!(buffers, (preview.rt_samples.as_ptr(), preview.rt_work.as_ptr(),
        preview.rt_scratch.as_ptr(), preview.last_fft_norms.as_ptr()));

    let left = live_tone(1024, 96_000);
    let right: Vec<_> = left.iter().map(|&v| -v).collect();
    preview.process_channels(&left, &right, 1.0 / 60.0);
    fft.process_channels(&left, &right, 1.0 / 60.0);
    assert_eq!(preview.bars_left, fft.bars_left);
    assert_eq!(preview.bars_right, fft.bars_right);
    assert_eq!(preview.take_channel_ffts(), 2);

    // Public live entry points cannot overwrite a CQT once it is installed.
    let stored = cache::PreFrames::from_v4_codes(vec![1234; 2 * BARS], 2, BARS);
    preview.set_pre_frames_with(stored, PreChannels::Missing, None, 180.0);
    preview.snap_pre_to(0.0);
    let cached = preview.magnitudes.clone();
    preview.process_realtime(1.0 / 60.0);
    preview.process_channels(&left, &right, 1.0 / 60.0);
    assert_eq!(preview.magnitudes, cached);
    assert!(preview.last_fft_norms.is_empty());
    assert_eq!(preview.take_channel_ffts(), 0);
}

#[test]
fn cqt_preview_uses_live_rate_endpoint_and_presentation_weights() {
    let _calib = gpu_calib::isolated_test_env("true_cqt_preview_axis");
    let f = Fixture::new("preview_axis");
    let mut w = window(&f.cache);
    w.analyzer.sample_rate = 44_100;
    w.analyzer.live_rate = 96_000;
    w.live_tap = Some(96_000);
    w.max_freq = 24_000.0;
    w.loudness_mode = LoudnessMode::EqualLoudness;
    w.analyzer.aslt_cfg.scale = freq_scale::FreqScale::Erb;
    w.sync_params();
    assert_eq!(w.display_sample_rate(), 96_000);
    assert_eq!(w.analysis_max_freq(), 24_000.0);
    assert_eq!(w.analyzer.eq_weights,
        compute_eq_weights(BARS, 220.0, 24_000.0, freq_scale::FreqScale::Log));
    // Previewing does not alter the eventual CQT identity or saved ASLT scale.
    let identity = w.analyzer.effective_settings(&f.track).canonical();
    w.analyzer.fft_size = 256;
    w.analyzer.rebuild_fft();
    *w.sample_buf.lock().unwrap() = live_tone(1024, 96_000);
    w.analyzer.tick_pre(0.0, 1.0 / 60.0);
    assert_eq!(w.analyzer.effective_settings(&f.track).canonical(), identity);
    assert_eq!(w.analyzer.aslt_cfg.scale, freq_scale::FreqScale::Erb);
    assert!(w.spectrum_source_note().unwrap().contains("96000 Hz live PCM"));

    // With a lower-rate live tap both the preview plan and its axis clamp.
    w.analyzer.live_rate = 32_000;
    w.live_tap = Some(32_000);
    w.sync_params();
    assert_eq!(w.analysis_max_freq(), 16_000.0);
    assert_eq!(w.analyzer.live_mapping_max_freq(32_000), 16_000.0);
    assert_eq!(w.analyzer.eq_weights,
        compute_eq_weights(BARS, 220.0, 16_000.0, freq_scale::FreqScale::Log));
}

#[test]
fn pending_cqt_grid_survives_wholetrack_ui_toggles_and_hidden_route_changes() {
    let _calib = gpu_calib::isolated_test_env("true_cqt_hidden_preview_grid");
    let f = Fixture::new("hidden_preview_grid");
    let mut w = window(&f.cache);
    w.open = true;
    w.whole_track.open = true;
    w.analyzer.sample_rate = 44_100;
    w.analyzer.live_rate = 96_000;
    w.live_tap = Some(96_000);
    w.max_freq = 24_000.0;
    w.loudness_mode = LoudnessMode::EqualLoudness;
    w.sync_params();
    assert!(w.whole_track_shown());
    assert_eq!(w.display_sample_rate(), 96_000);
    assert_eq!(w.analysis_max_freq(), 24_000.0);
    assert_eq!(w.analyzer.eq_weights,
        compute_eq_weights(BARS, 220.0, 24_000.0, freq_scale::FreqScale::Log));

    let mut ui = HeadlessUi::new();
    ui.click_label(&mut w, "Whole track");
    assert!(!w.whole_track.open, "the real Whole track toggle did not close");
    assert!(w.true_cqt_live_preview());
    assert_eq!(w.display_sample_rate(), 96_000);
    assert_eq!(w.analyzer.eq_weights,
        compute_eq_weights(BARS, 220.0, 24_000.0, freq_scale::FreqScale::Log),
        "closing Whole track exposed ISO weights for the analysis-rate grid");

    ui.click_label(&mut w, "Whole track");
    assert!(w.whole_track_shown());
    // A route restart while the preview is hidden must update its weights too.
    w.set_live_route(Some(32_000));
    assert_eq!(w.analyzer.sample_rate, 44_100);
    assert_eq!(w.display_sample_rate(), 32_000);
    assert_eq!(w.analysis_max_freq(), 16_000.0);
    assert_eq!(w.analyzer.eq_weights,
        compute_eq_weights(BARS, 220.0, 16_000.0, freq_scale::FreqScale::Log));
    assert!(w.analyzer.last_fft_norms.is_empty());
    assert!(wholetrack::Snapshot::capture(&w.analyzer, false).is_err());
    ui.click_label(&mut w, "Whole track");
    assert!(!w.whole_track.open);
    assert!(w.true_cqt_live_preview());
    assert_eq!(w.display_sample_rate(), 32_000);
    assert_eq!(w.analyzer.eq_weights,
        compute_eq_weights(BARS, 220.0, 16_000.0, freq_scale::FreqScale::Log));
    assert_eq!(spawned(&w.analyzer), 0, "a viewer toggle started an analysis");
    assert_eq!(f.heads(), 0);
}

#[test]
fn a_missing_pcm_route_clears_preview_and_wholetrack_never_uses_it() {
    let _calib = gpu_calib::isolated_test_env("true_cqt_preview_no_pcm");
    let f = Fixture::new("preview_no_pcm");
    let mut w = window(&f.cache);
    w.live_tap = None;
    w.analyzer.last_fft_norms = vec![1.0; 8];
    w.analyzer.magnitudes = vec![1.0; BARS];
    *w.sample_buf.lock().unwrap() = live_tone(16_384, SR);
    w.stereo_buf.lock().unwrap().channels = 2;
    w.tick(0.0, true);
    assert!(w.analyzer.last_fft_norms.is_empty());
    assert!(w.analyzer.magnitudes.iter().all(|&v| v == 0.0));
    assert_eq!(w.channel_availability, channels::ChannelAvailability::NoLiveTap);
    assert!(w.spectrum_source_note().unwrap().contains("unavailable"));

    w.open = true;
    w.whole_track.open = true;
    w.live_tap = Some(SR);
    w.last_fft_time = None;
    w.tick(0.0, true);
    assert!(wholetrack::Snapshot::capture(&w.analyzer, false).is_err());
    assert!(w.analyzer.last_fft_norms.is_empty());
    assert!(w.spectrum_source_note().unwrap().contains("whole-track view is waiting"));
}

#[test]
fn a_real_cqt_arrival_replaces_the_labelled_preview_and_its_waterfall() {
    let _calib = gpu_calib::isolated_test_env("true_cqt_preview_arrival");
    let f = Fixture::new("preview_arrival");
    let mut w = window(&f.cache);
    w.open = true;
    w.style = VizStyle::Waterfall;
    // Keep the production worker before its first operation: the pre-cache
    // state cannot depend on whether the machine happens to finish quickly.
    let held = ReleaseHold(Arc::new(AtomicBool::new(true)));
    w.analyzer.analysis_hold = Some(held.0.clone());
    w.on_play_live(&f.track, SR, Some(96_000));
    assert_eq!(spawned(&w.analyzer), 1);
    w.analyzer.fft_size = 256;
    w.analyzer.rebuild_fft();
    *w.sample_buf.lock().unwrap() = live_tone(1024, 96_000);
    for frame in 0..3 {
        w.last_fft_time = None;
        w.tick(frame as f64 / 180.0, true);
    }
    assert!(w.analyzer.pre_frames.is_empty());
    assert!(w.analyzer.pre_record().is_none());
    assert!(w.analyzer.pre_cache().is_none());
    assert_eq!(f.heads(), 0);
    assert!(wholetrack::Snapshot::capture(&w.analyzer, false).is_err());
    assert!(w.analyzer.last_fft_norms.iter().any(|&v| v > 0.0));
    assert!(w.analyzer.magnitudes.iter().any(|&v| v > 0.0));
    assert!(w.analyzer.waterfall.iter().any(|r| r.iter().any(|&v| v > 0.0)));
    let mut ui = HeadlessUi::new();
    let labels = ui.frame(&mut w, Vec::new());
    assert!(labels.iter().any(|(s, _)| s.starts_with("LIVE FFT PREVIEW")),
        "the actual plot did not label its temporary FFT source");
    assert!(w.spectrum_source_note().unwrap().contains("96000 Hz live PCM"));

    w.analyzer.analysis_hold = None;
    drop(held);
    let deadline = Instant::now() + Duration::from_secs(60);
    // Do not consume the result here: the next ordinary window tick must
    // install it, reset the old presentation, and draw the new source.
    while w.analyzer.is_analyzing.load(Ordering::Relaxed) {
        assert!(Instant::now() < deadline, "held CQT worker did not finish");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(w.analyzer.pre_frames.is_empty());
    w.last_fft_time = None;
    w.tick(0.1, true);
    settle(&mut w.analyzer);
    assert_eq!(w.analyzer.pre_frame_rate, 180.0);
    assert_eq!(w.bar_mapping, BarMappingMode::TrueCqt(config()));
    assert!(w.analyzer.pre_record().unwrap().contains("map=true-cqt"));
    assert!(w.analyzer.last_fft_norms.is_empty());
    assert!(w.analyzer.magnitudes.iter().all(|&v| v == 0.0),
        "anti-phase CQT Mix retained the nonzero FFT preview");
    assert!(!w.analyzer.waterfall.is_empty());
    assert!(w.analyzer.waterfall.iter().all(|r| r.iter().all(|&v| v == 0.0)),
        "the live FFT waterfall survived the producer change");
    assert_eq!(w.display_sample_rate(), SR);
    assert_eq!(f.heads(), 1);
    assert!(wholetrack::Snapshot::capture(&w.analyzer, false).is_ok());
    let labels = ui.frame(&mut w, Vec::new());
    assert!(labels.iter().any(|(s, _)| s.starts_with("TRUE CQT · pre-processed analysis")));
    assert!(!labels.iter().any(|(s, _)| s.starts_with("LIVE FFT PREVIEW")));
}

#[test]
fn production_cqt_stereo_analysis_reload_and_repeat_keep_the_cqt_clock() {
    let _calib = gpu_calib::isolated_test_env("true_cqt_production");
    let f = Fixture::new("production");
    let mut w = window(&f.cache);
    w.on_play(&f.track, SR);
    assert_eq!(spawned(&w.analyzer), 1);
    settle(&mut w.analyzer);
    assert_eq!(w.analyzer.pre_frame_rate, 180.0);
    assert_eq!(w.analyzer.pre_frames.len(), 60);
    assert_eq!(w.analyzer.pre_frames.bars(), BARS);
    for frame in 0..w.analyzer.pre_frames.len() {
        assert!(
            w.analyzer
                .pre_frames
                .row(frame)
                .unwrap()
                .iter()
                .all(|&c| c == 0),
            "anti-phase Mix was made by averaging magnitudes"
        );
    }
    let pair = w
        .analyzer
        .pre_stereo()
        .expect("fresh joint CQT lost its channels");
    assert!(pair.left().row(30).unwrap().iter().any(|&c| c > 0));
    assert_eq!(
        pair.left(),
        pair.right(),
        "opposite polarities changed channel magnitude"
    );
    assert_eq!(
        f.heads(),
        1,
        "successful CQT did not publish exactly one head"
    );

    let record = w
        .analyzer
        .pre_record()
        .expect("installed CQT lacks its record");
    assert!(record.contains("map=true-cqt"));
    assert!(record.contains("hop=variable"));
    let geometry = wholetrack::Geometry::parse(record).expect("CQT geometry was not readable");
    assert_eq!(geometry.frame_rate(), 180.0);
    assert_eq!(geometry.frame_time(30.0), 30.0 / 180.0);
    assert_eq!(
        geometry.transform,
        wholetrack::Transform::TrueCqt { config: config() }
    );
    let snapshot = wholetrack::Snapshot::capture(&w.analyzer, true).unwrap();
    assert!(snapshot.lr.is_some());

    let fresh = w.analyzer.pre_frames.clone();
    let revision = w.analyzer.pre_revision();
    w.on_play(&f.track, SR);
    assert!(
        w.analyzer.analysis_idle(),
        "repeat started another analysis"
    );
    assert_eq!(spawned(&w.analyzer), 1);
    assert!(Arc::ptr_eq(&fresh, &w.analyzer.pre_frames));
    assert_eq!(w.analyzer.pre_revision(), revision);
    assert_eq!(w.analyzer.pre_frame_rate, 180.0);

    let mut reload = window(&f.cache);
    reload.on_play(&f.track, SR);
    assert!(
        reload.analyzer.analysis_idle(),
        "exact CQT hit started a worker"
    );
    assert_eq!(spawned(&reload.analyzer), 0);
    assert_eq!(reload.analyzer.pre_frame_rate, 180.0);
    assert_eq!(*reload.analyzer.pre_frames, *fresh);
    assert!(
        reload.analyzer.has_pre_channels(),
        "reload did not discover stored CQT channels"
    );
}

struct ReleaseHold(Arc<AtomicBool>);

impl Drop for ReleaseHold {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

#[test]
fn a_cqt_cancelled_before_work_publishes_nothing_and_can_retry() {
    let _calib = gpu_calib::isolated_test_env("true_cqt_cancel");
    let f = Fixture::new("cancel");
    let mut w = window(&f.cache);
    let held = ReleaseHold(Arc::new(AtomicBool::new(true)));
    w.analyzer.analysis_hold = Some(held.0.clone());
    w.on_play(&f.track, SR);
    assert_eq!(spawned(&w.analyzer), 1);
    w.analyzer.abort_preprocess();
    assert!(w.analyzer.abort_analysis.load(Ordering::SeqCst));
    w.analyzer.analysis_hold = None;
    drop(held);
    settle(&mut w.analyzer);
    assert!(w.analyzer.pre_frames.is_empty());
    assert_eq!(f.heads(), 0, "cancelled CQT published a head");
    {
        let outcomes = w.analyzer.lifecycle_probe.outcomes.lock().unwrap();
        assert_eq!(
            outcomes.len(),
            1,
            "cancel test did not observe a worker result"
        );
        assert_eq!(outcomes[0].kind, "aborted");
    }

    w.analyzer.start_preprocess(f.track.clone());
    settle(&mut w.analyzer);
    assert!(
        !w.analyzer.pre_frames.is_empty(),
        "cancelled slot prevented retry"
    );
    assert_eq!(f.heads(), 1);
}

#[test]
fn a_full_support_cqt_refusal_publishes_no_partial_head() {
    let _calib = gpu_calib::isolated_test_env("true_cqt_refusal");
    let f = Fixture::new("refusal");
    let mut w = window(&f.cache);
    let demanding = cqt::Config {
        bins_per_octave: 192,
        ..config()
    };
    w.bar_mapping = BarMappingMode::TrueCqt(demanding);
    w.analyzer.bar_mapping = w.bar_mapping.clone();
    w.min_freq = 1.0;
    w.analyzer.min_freq = 1.0;
    w.on_play(&f.track, SR);
    settle(&mut w.analyzer);
    assert!(
        w.analyzer.pre_frames.is_empty(),
        "oversized support silently fell back"
    );
    assert_eq!(f.heads(), 0, "refused CQT published a partial head");
    let outcomes = w.analyzer.lifecycle_probe.outcomes.lock().unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].kind, "error");
}

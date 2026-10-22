//! The wider UI capacity does not change default settings, cache identities,
//! or the reader's allocation envelope. Fixtures never use the owner's files.

use super::*;

const BARS: usize = 2048;

fn codes(frames: usize, bars: usize, salt: usize) -> cache::PreFrames {
    let values = (0..frames * bars)
        .map(|i| ((i * 37 + (i / bars) * 19 + salt) & 4095) as u16)
        .collect();
    cache::PreFrames::from_v4_codes(values, frames, bars)
}

#[test]
fn wider_bar_profiles_and_slider_snapping_preserve_the_default() {
    let _calib = gpu_calib::isolated_test_env("bar_capacity_settings");
    assert_eq!(DEFAULT_BAR_COUNT, 1024);
    assert_eq!(MAX_BAR_COUNT, BARS);
    let mut window = SpectrumWindow::new_fresh_profile();
    assert_eq!(window.bar_count, DEFAULT_BAR_COUNT);
    let mut settings = window.snapshot();
    for (stored, expected) in [(0, 16), (1024, 1024), (2048, 2048), (4096, 2048)] {
        settings.bar_count = stored;
        window.apply_settings(&settings);
        assert_eq!(window.bar_count, expected);
        assert_eq!(window.analyzer.bar_count, expected);
        let json = serde_json::to_string(&window.snapshot()).unwrap();
        let reloaded = SpectrumWindow::new_from_settings_json(&json);
        assert_eq!(reloaded.bar_count, expected);
        assert_eq!(reloaded.analyzer.bar_count, expected);
    }
    for (value, expected) in [
        (1024, 1024),
        (1535, 1024),
        (1536, 2048),
        (2048, 2048),
        (4096, 2048),
    ] {
        assert_eq!(snap_pow2(value), expected);
    }
}

#[test]
fn wider_mix_and_channel_code_planes_round_trip_exactly() {
    for bars in [1024, BARS] {
        let mono = codes(7, bars, 11);
        let pair = stereo::Stereo::new(codes(7, bars, 53), codes(7, bars, 97)).unwrap();
        let record = format!("bar-capacity-regression;bars={bars}");
        let mix_bytes = mono.to_v4_with(record.as_bytes());
        let decoded = cache::decode(&mix_bytes, bars, MAX_BAR_COUNT).unwrap();
        assert_eq!(decoded, mono);
        assert_eq!(cache::record_of(&mix_bytes), Some(record.as_str()));
        let id = stereo::Identity {
            source: 0x1234,
            params: bars as u64,
        };
        let sidecar = stereo::encode(id, &mono, &pair);
        let channels = stereo::decode_with(
            &sidecar,
            id,
            &decoded,
            MAX_BAR_COUNT,
            cache::Limits::default(),
        )
        .unwrap();
        assert_eq!(channels, pair);
        assert!(pair.left().row(0) != pair.right().row(0));
    }
}

#[test]
fn wider_cache_headers_do_not_expand_the_original_cell_budget() {
    let limits = cache::Limits::default();
    assert_eq!(limits.max_cells, 512_000_000);
    assert_eq!(limits.max_cells, cache::MAX_DECODED_CELLS);
    assert_eq!(limits.max_file_bytes, 512 << 20);
    assert_eq!(limits.max_expansion, 1024);
    let frames = limits.max_cells / BARS + 1;
    assert!(frames < cache::MAX_FRAMES);
    let mut header = Vec::new();
    header.extend_from_slice(&0x4D53_5034u32.to_le_bytes());
    header.extend_from_slice(&(frames as u32).to_le_bytes());
    header.extend_from_slice(&(BARS as u32).to_le_bytes());
    assert!(matches!(
        cache::decode_with(&header, BARS, MAX_BAR_COUNT, limits),
        Err(cache::CacheError::TooLarge { what: "decoded cells", got, limit })
            if got == frames * BARS && limit == 512_000_000
    ));
}

#[test]
fn producer_dimensions_are_refused_before_work_not_after_publication() {
    assert!(validate_preprocess_capacity(250_000, 2048).is_ok());
    assert!(validate_preprocess_capacity(250_001, 2048).is_err());
    assert!(validate_preprocess_capacity(500_000, 1024).is_ok());
    assert!(validate_preprocess_capacity(500_001, 1024).is_err());
    assert!(validate_preprocess_capacity(0, 2048).is_err());
    assert!(validate_preprocess_capacity(100, 2049).is_err());
    assert!(validate_preprocess_capacity(usize::MAX, 2048).is_err());
}

#[test]
fn live_mix_channels_and_waterfall_keep_all_2048_values() {
    let sample_rate = 48_000;
    let size = 1024;
    let tone = |hz: f32| -> Vec<f32> {
        (0..size * 2)
            .map(|i| (std::f32::consts::TAU * hz * i as f32 / sample_rate as f32).sin() * 0.5)
            .collect()
    };
    let left = tone(1500.0);
    let right = tone(6000.0);
    let mappings = [
        BarMappingMode::FlatOverlap,
        BarMappingMode::Gaussian,
        BarMappingMode::Cqt,
        BarMappingMode::Superlet,
        BarMappingMode::TrueCqt(cqt::Config::default()),
    ];
    for mapping in mappings {
        let samples = new_sample_buf();
        samples
            .lock()
            .unwrap()
            .extend(left.iter().zip(&right).map(|(&l, &r)| (l + r) * 0.5));
        let mut analyzer = SpectrumAnalyzer::new(samples);
        analyzer.fft_size = size;
        analyzer.pre_fft_size = size;
        analyzer.live_rate = sample_rate;
        analyzer.sample_rate = sample_rate;
        analyzer.bar_mapping = mapping.clone();
        analyzer.smoothing = 0.0;
        analyzer.set_bar_count(BARS);
        analyzer.rebuild_fft();
        analyzer.waterfall_enabled = true;
        for _ in 0..2 {
            analyzer.process_realtime(1.0 / 180.0);
            analyzer.process_channels(&left, &right, 1.0 / 180.0);
            assert_eq!(analyzer.magnitudes.len(), BARS, "{mapping:?}");
            assert_eq!(analyzer.peak_input.len(), BARS, "{mapping:?}");
            assert_eq!(analyzer.bars_left.len(), BARS, "{mapping:?}");
            assert_eq!(analyzer.bars_right.len(), BARS, "{mapping:?}");
            for values in [
                &analyzer.magnitudes,
                &analyzer.bars_left,
                &analyzer.bars_right,
            ] {
                assert!(
                    values
                        .iter()
                        .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                );
            }
            assert!(analyzer.bars_left != analyzer.bars_right);
        }
        assert_eq!(analyzer.waterfall.len(), 2);
        assert!(analyzer.waterfall.iter().all(|row| row.len() == BARS));
        assert_eq!(analyzer.channel_frame().mix.len(), BARS);
        assert_eq!(analyzer.channel_frame().left.unwrap().len(), BARS);
        assert_eq!(analyzer.channel_frame().right.unwrap().len(), BARS);
    }
}

struct Fixture {
    root: PathBuf,
    temp: PathBuf,
    track: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let temp = std::fs::canonicalize(std::env::temp_dir()).unwrap();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let requested = temp.join(format!(
            "moosik_bar_capacity_{}_{nonce:x}",
            std::process::id()
        ));
        std::fs::create_dir(&requested).unwrap();
        let root = std::fs::canonicalize(&requested).unwrap();
        assert_eq!(root.parent(), Some(temp.as_path()));
        let track = root.join("short-stereo.wav");
        let mut data = Vec::new();
        for i in 0..4096 {
            for hz in [1500.0, 6000.0] {
                let value = (std::f64::consts::TAU * hz * i as f64 / 16_000.0).sin();
                data.extend_from_slice(&((value * 12000.0) as i16).to_le_bytes());
            }
        }
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&((data.len() + 36) as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16_000u32.to_le_bytes());
        wav.extend_from_slice(&64_000u32.to_le_bytes());
        wav.extend_from_slice(&4u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);
        std::fs::write(&track, wav).unwrap();
        Self { root, temp, track }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if self.root.parent() == Some(self.temp.as_path()) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }
}

#[test]
fn all_preprocess_methods_publish_and_reload_real_2048_bar_channels() {
    let _calib = gpu_calib::isolated_test_env("bar_capacity_production");
    gpu_calib::set_mode(gpu_calib::GpuMode::Off);
    let fixture = Fixture::new();
    let sample_rate = 16_000;
    let fft_size = 1024;
    let aslt = aslt::AsltConfig {
        q_ratio: 1.0,
        max_window_s: 0.025,
        n_wavelets: 1,
        spread: 0.5,
        scale: freq_scale::FreqScale::Log,
    };
    let mappings = [
        BarMappingMode::FlatOverlap,
        BarMappingMode::Cqt,
        BarMappingMode::Superlet,
        BarMappingMode::TrueCqt(cqt::Config {
            bins_per_octave: 12,
            window: cqt::Window::Hann,
            frame_rate: 30.0,
            bass_limit_s: None,
        }),
    ];
    for (index, mapping) in mappings.iter().enumerate() {
        let unit = fixture.root.join(format!("method-{index}.spectrumcache"));
        let store = test_store(&unit).unwrap();
        let settings = settings_for(
            &fixture.track,
            BARS,
            fft_size,
            2,
            0.5,
            &WindowFn::Hann,
            1000.0,
            7000.0,
            mapping,
            &InterpolationMode::None,
            crate::dsd::decimate::LEGACY_ANALYSIS_RATE,
            &aslt,
            30.0,
        );
        let result = preprocess_file(
            &fixture.track,
            &unit,
            Some(store.clone()),
            true,
            sample_rate,
            BARS,
            &Arc::new(AtomicUsize::new(0)),
            fft_size,
            2,
            0.5,
            &WindowFn::Hann,
            1000.0,
            7000.0,
            &InterpolationMode::None,
            mapping,
            crate::dsd::decimate::LEGACY_ANALYSIS_RATE,
            &aslt,
            30.0,
            &Arc::new(AtomicBool::new(false)),
            &Arc::new(AtomicUsize::new(usize::MAX)),
            StereoRequest::Joint,
        );
        let (frames, channels, meta) = match result {
            PreMessage::Done {
                frames,
                stereo: Some(pair),
                meta,
                ..
            } => (frames, pair, meta),
            PreMessage::Error(error) => panic!("{mapping:?}: {error}"),
            _ => panic!("{mapping:?}: did not complete its joint analysis"),
        };
        assert_eq!(frames.bars(), BARS);
        assert!(!frames.is_empty());
        assert_eq!((channels.len(), channels.bars()), (frames.len(), BARS));
        assert!(channels.left().row(0) != channels.right().row(0));
        assert!(
            meta.published.is_some(),
            "{mapping:?}: no generation published"
        );
        let descriptor = store::Descriptor {
            record: settings.record(sample_rate),
            source: meta.source.clone(),
        };
        let (loaded, installed, _) =
            load_generation(&store, &settings.stem(), &descriptor, BARS).unwrap();
        assert_eq!(
            loaded, frames,
            "{mapping:?}: actual Mix readback changed codes"
        );
        let payload = store.read_lr(&installed).unwrap();
        let loaded_pair = stereo::decode_with(
            &payload,
            meta.identity,
            &loaded,
            MAX_BAR_COUNT,
            cache::Limits::default(),
        )
        .unwrap();
        assert_eq!(
            loaded_pair, *channels,
            "{mapping:?}: actual L/R readback changed codes"
        );
        let geometry = wholetrack::Geometry::parse(&descriptor.record).unwrap();
        assert_eq!(geometry.data_bars, BARS);
    }
}

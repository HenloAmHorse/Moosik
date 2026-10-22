//! Actual compressed input through the production analysis/cache publication.
use super::*;

#[test]
fn common_audio_formats_publish_mix_and_channel_analysis() {
    let _calibration = gpu_calib::isolated_test_env("audio_import");
    gpu_calib::set_mode(gpu_calib::GpuMode::Off);
    let temp = std::fs::canonicalize(std::env::temp_dir()).unwrap();
    let root = temp.join(format!(
        "moosik-format-analysis-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    for (index, name) in [
        "tone.opus",
        "tone.webm",
        "tone-alac.m4a",
        "tone-aac.m4a",
        "tone.aiff",
        "tone.caf",
    ]
    .iter()
    .enumerate()
    {
        let track = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/testdata/audio")
            .join(name);
        let cache = root.join(index.to_string()).join("test.spectrumcache");
        let cfg = aslt::AsltConfig {
            n_wavelets: 1,
            max_window_s: 0.25,
            ..aslt::AsltConfig::default()
        };
        let progress = Arc::new(AtomicUsize::new(0));
        let abort = Arc::new(AtomicBool::new(false));
        let eta = Arc::new(AtomicUsize::new(usize::MAX));
        let mapping = if index == 0 {
            BarMappingMode::Superlet
        } else {
            BarMappingMode::TrueCqt(cqt::Config {
                bins_per_octave: 12,
                window: cqt::Window::Blackman,
                frame_rate: 180.0,
                bass_limit_s: None,
            })
        };
        let message = preprocess_file(
            &track,
            &cache,
            test_store(&cache),
            true,
            0,
            16,
            &progress,
            DEFAULT_FFT_SIZE,
            1,
            0.875,
            &WindowFn::Hann,
            100.0,
            12_000.0,
            &InterpolationMode::None,
            &mapping,
            88_200,
            &cfg,
            180.0,
            &abort,
            &eta,
            StereoRequest::Joint,
        );
        let PreMessage::Done {
            frames,
            stereo,
            meta,
            ..
        } = message
        else {
            panic!("{name}: analysis did not complete")
        };
        assert!(!frames.is_empty(), "{name}");
        let pair = stereo.unwrap();
        assert_eq!(pair.left().len(), frames.len());
        assert!(meta.published.is_some(), "{name}: not published");
        let files = published_heads(&cache);
        assert_eq!(files.len(), 1);
        assert!(files[0].1.1.is_some());
        drop(meta);
    }
    // This exact, freshly created direct child is the only deletion target.
    let resolved = std::fs::canonicalize(&root).unwrap();
    assert_eq!(resolved.parent(), Some(temp.as_path()));
    std::fs::remove_dir_all(resolved).unwrap();
}

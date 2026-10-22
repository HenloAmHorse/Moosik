//! Explicit live padding: the actual FFT, all consumers, and legacy settings.
use super::*;

fn pcm(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = i as f32;
            0.32 * (x * 0.019 + phase).sin()
                + 0.09 * (x * 0.47 - phase).cos()
                + ((i * 7919 % 65521) as f32 / 65521.0 - 0.5) * 0.04
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|v| v.to_bits()).collect()
}
fn complex_bits(v: &[Complex<f32>]) -> Vec<(u32, u32)> {
    v.iter().map(|v| (v.re.to_bits(), v.im.to_bits())).collect()
}

#[test]
fn every_live_padding_factor_matches_an_explicit_padded_fft_for_mix_and_channels() {
    let mut a = SpectrumAnalyzer::new(new_sample_buf());
    a.rt_disable_input_reuse = true;
    for size in [1024, 4096, 8192] {
        for pad in [8, 1, 4, 2] {
            // downsize too: stale padding must not survive.
            for window in [
                WindowFn::Hann,
                WindowFn::Hamming,
                WindowFn::Blackman,
                WindowFn::FlatTop,
            ] {
                a.fft_size = size;
                a.rt_pad_factor = pad;
                a.window_fn = window;
                a.rebuild_realtime_fft();
                let padded = size * pad;
                let plan = FftPlanner::new().plan_fft_forward(padded);
                for tick in 0..3 {
                    let source = if tick == 2 {
                        vec![0.0; size + 17]
                    } else {
                        pcm(size + 17, tick as f32)
                    };
                    let mut expected: Vec<_> = source[source.len() - size..]
                        .iter()
                        .zip(&a.window_coeffs)
                        .map(|(x, w)| Complex::new(x * w, 0.0))
                        .chain(std::iter::repeat_n(Complex::default(), padded - size))
                        .collect();
                    plan.process(&mut expected);
                    let norms: Vec<_> = expected[..padded / 2].iter().map(|v| v.norm()).collect();
                    *a.sample_buf.lock().unwrap() = source.clone();
                    assert!(a.transform_realtime());
                    assert_eq!(
                        complex_bits(&a.rt_work),
                        complex_bits(&expected),
                        "size={size} pad={pad} tick={tick}"
                    );
                    assert_eq!(bits(&a.last_fft_norms), bits(&norms));
                    a.transform_channel(1, &source);
                    assert_eq!(complex_bits(&a.ch_scratch), complex_bits(&expected));
                    assert_eq!(bits(&a.ch_norms[1]), bits(&norms));
                    assert_eq!(bits(&a.last_fft_norms), bits(&norms));
                    let count = crate::alloc_probe::arm();
                    assert!(a.transform_realtime());
                    a.transform_channel(1, &source);
                    assert_eq!(crate::alloc_probe::disarm(count), 0);
                }
            }
        }
    }
}

#[test]
fn padded_ticks_keep_all_display_rows_equal_to_the_frozen_mapping() {
    for pad in [1, 2, 4, 8] {
        for mapping in [
            BarMappingMode::FlatOverlap,
            BarMappingMode::Gaussian,
            BarMappingMode::Cqt,
        ] {
            for bars in [1024, 2048] {
                let set = |a: &mut SpectrumAnalyzer| {
                    a.fft_size = 4096;
                    a.rt_pad_factor = pad;
                    a.bar_count = bars;
                    a.live_rate = 48_000;
                    a.sample_rate = 48_000;
                    a.aslt_cfg.scale = freq_scale::FreqScale::Blend(0.5);
                    a.bar_mapping = mapping.clone();
                };
                let mut prod = realtime_route_tests::rt_analyzer(&set, false);
                let mut old = realtime_route_tests::rt_analyzer(&set, true);
                realtime_route_tests::drive_pair("padded mapping", &mut prod, &mut old, 4);
                let l = pcm(5000, 0.0);
                let r = pcm(5000, 0.7);
                prod.process_channels(&l, &r, 1.0 / 180.0);
                old.process_channels(&l, &r, 1.0 / 180.0);
                assert_eq!(bits(&prod.bars_left), bits(&old.bars_left));
                assert_eq!(bits(&prod.bars_right), bits(&old.bars_right));
                assert_eq!(prod.last_fft_norms.len(), prod.fft_size * pad / 2);
            }
        }
    }
}

#[test]
fn identical_pcm_reuses_only_the_fft_not_smoothing_peaks_or_history() {
    let set = |a: &mut SpectrumAnalyzer| {
        a.fft_size = 4096;
        a.rt_pad_factor = 8;
        a.bar_count = 2048;
        a.bar_mapping = BarMappingMode::Gaussian;
    };
    let mut new = realtime_route_tests::rt_analyzer(&set, false);
    let mut old = realtime_route_tests::rt_analyzer(&set, false);
    old.rt_disable_input_reuse = true;
    let right = pcm(4096, 1.7);
    for step in 0..12 {
        let left = pcm(4096, (step / 4) as f32 * 0.3);
        for a in [&mut new, &mut old] {
            *a.sample_buf.lock().unwrap() = left.clone();
            a.process_realtime(1.0 / 180.0);
            a.process_channels(&left, &right, 1.0 / 180.0);
        }
        assert_eq!(
            realtime_route_tests::state(&new),
            realtime_route_tests::state(&old)
        );
        assert_eq!(bits(&new.bars_left), bits(&old.bars_left));
        assert_eq!(bits(&new.bars_right), bits(&old.bars_right));
    }
    assert_eq!((new.rt_fft_computations, new.rt_fft_reuses), (3, 9));
    assert_eq!((old.rt_fft_computations, old.rt_fft_reuses), (12, 0));
    // Left repeats like the Mix (9); held right repeats after its first (11).
    assert_eq!((new.ch_fft_reuses, old.ch_fft_reuses), (20, 0));
    assert_eq!(
        new.waterfall.len(),
        12,
        "held PCM must still advance history"
    );
}

#[test]
fn fft_reuse_requires_all_input_bits_a_current_snapshot_and_current_norms() {
    let mut a = SpectrumAnalyzer::new(new_sample_buf());
    a.fft_size = 1024;
    a.rt_pad_factor = 4;
    a.rebuild_realtime_fft();
    let mut x = pcm(1024, 0.4);
    x[0] = 0.0;
    *a.sample_buf.lock().unwrap() = x.clone();
    assert!(a.transform_realtime());
    assert!(a.transform_realtime());
    assert_eq!((a.rt_fft_computations, a.rt_fft_reuses), (1, 1));
    x[0] = -0.0;
    *a.sample_buf.lock().unwrap() = x;
    assert!(a.transform_realtime());
    assert_eq!(
        a.rt_fft_computations, 2,
        "+0 and -0 input bits were conflated"
    );
    a.window_fn = WindowFn::Blackman;
    a.rebuild_realtime_fft();
    assert!(a.transform_realtime());
    assert_eq!(
        a.rt_fft_computations, 3,
        "empty snapshot reused an old window"
    );
    a.last_fft_norms.clear();
    assert!(a.transform_realtime());
    assert_eq!(a.rt_fft_computations, 4);
    a.sample_buf.lock().unwrap().clear();
    assert!(!a.transform_realtime());
    assert_eq!((a.rt_fft_computations, a.rt_fft_reuses), (4, 1));
}

#[test]
fn live_padding_migrates_without_changing_preprocess_padding_or_saved_responsive_limits() {
    let _env = gpu_calib::isolated_test_env("live_pad_profiles");
    let old: SpectrumSettings = serde_json::from_str(r#"{"pad_factor":16}"#).unwrap();
    assert_eq!(old.rt_pad_factor, 2);
    let mut w = SpectrumWindow::new_from_settings_json(r#"{"pad_factor":16,"rt_pad_factor":8}"#);
    assert_eq!(
        (
            w.pad_factor,
            w.analyzer.pad_factor,
            w.analyzer.rt_pad_factor
        ),
        (16, 16, 8)
    );
    let before = w.snapshot();
    let cache = Arc::clone(&w.analyzer.pre_frames);
    let background_plan = Arc::clone(&w.analyzer.fft_plan);
    w.set_realtime_padding(4);
    assert_eq!(w.analyzer.rt_fft_plan.len(), w.analyzer.fft_size * 4);
    assert!(Arc::ptr_eq(&background_plan, &w.analyzer.fft_plan));
    assert!(Arc::ptr_eq(&cache, &w.analyzer.pre_frames));
    let mut after = w.snapshot();
    assert_eq!(after.rt_pad_factor, 4);
    after.rt_pad_factor = before.rt_pad_factor;
    assert!(
        after == before,
        "a live-only control changed analysis settings"
    );
    for limit in [0.25, 0.5, 0.83] {
        let raw = format!(r#"{{"bass_limit_s":{limit}}}"#);
        let config: cqt::Config = serde_json::from_str(&raw).unwrap();
        assert_eq!(config.bass_limit_s, Some(limit));
    }
    assert_eq!(cqt::DEFAULT_RESPONSIVE_LIMIT_S, 0.25);
    assert_eq!(cqt::Config::default().bass_limit_s, None);
    for invalid in [0, 3, 16, usize::MAX] {
        w.set_realtime_padding(invalid);
        assert_eq!(w.analyzer.rt_pad_factor, DEFAULT_REALTIME_PAD);
    }
}

#[test]
fn live_padding_does_not_blank_a_paused_cached_channel_view() {
    let _env = gpu_calib::isolated_test_env("live_pad_paused_channels");
    let mut w = SpectrumWindow::new_fresh_profile();
    w.mode = SpectrumMode::PreProcess;
    w.bar_mapping = BarMappingMode::Superlet;
    w.analyzer.bar_mapping = BarMappingMode::Superlet;
    w.channel_view = channels::ChannelView::Split;
    w.style = VizStyle::Bars;
    w.bar_count = 16;
    w.analyzer.set_bar_count(16);
    let plane = |v| cache::PreFrames::from_analysis(&vec![vec![v; 16]; 3]);
    let pair = stereo::Stereo::new(plane(0.3), plane(0.8)).unwrap();
    w.analyzer
        .set_pre_frames_with(plane(0.5), PreChannels::Resident(pair), None, 180.0);
    w.tick(0.01, false);
    assert_eq!(w.analyzer.bars_left.len(), 16);
    assert_ne!(w.analyzer.bars_left, w.analyzer.bars_right);
    let before = (
        bits(&w.analyzer.magnitudes),
        bits(&w.analyzer.bars_left),
        bits(&w.analyzer.bars_right),
    );
    w.set_realtime_padding(8);
    w.tick(0.01, false);
    assert_eq!(
        (
            bits(&w.analyzer.magnitudes),
            bits(&w.analyzer.bars_left),
            bits(&w.analyzer.bars_right)
        ),
        before
    );
}

#[test]
fn padding_changes_auxiliary_frequency_geometry_but_not_its_level_reference() {
    let size = 4096;
    let pal = Palette::new(SpectrumPalette::Classic, Color32::WHITE);
    let mut prior = None;
    // Uniform norms make the level independent of which bin is selected:
    // changing only the geometry must not dim the RTA or spectrogram.
    for pad in [2, 1, 4, 8] {
        let norms = vec![size as f32 * 0.1; size * pad / 2];
        let grid = LiveFftGeometry {
            padded: size * pad,
            level_reference: size * 2,
        };
        let bands = octave_band_magnitudes(&norms, 48000, grid);
        let mut spectrogram = Spectrogram::new();
        spectrogram.push_frame(&norms, 48000, grid, 20.0, 24000.0, &pal);
        let column: Vec<_> = (0..SPEC_H)
            .map(|row| spectrogram.pixels[row * SPEC_W])
            .collect();
        let current = (
            bands
                .iter()
                .map(|(f, v)| (f.to_bits(), v.to_bits()))
                .collect::<Vec<_>>(),
            column,
        );
        if let Some(prior) = &prior {
            assert_eq!(&current, prior);
        }
        prior = Some(current);
    }
}

#[test]
fn overlap_plans_follow_every_geometry_input_and_leave_dynamic_controls_dynamic() {
    type Edit = fn(&mut SpectrumAnalyzer);
    let changes: [Edit; 15] = [
        |a| a.bar_count = 2048,
        |a| a.live_rate = 44100,
        |a| a.min_freq = 35.0,
        |a| a.max_freq = 18000.0,
        |a| a.aslt_cfg.scale = freq_scale::FreqScale::Erb,
        |a| a.aslt_cfg.scale = freq_scale::FreqScale::Blend(0.4),
        |a| a.aslt_cfg.scale = freq_scale::FreqScale::from_parts(0.2, 900.0, 1.7, 2.5),
        |a| a.aslt_cfg.scale = freq_scale::FreqScale::from_parts(0.3, 900.0, 1.7, 2.5),
        |a| a.aslt_cfg.scale = freq_scale::FreqScale::from_parts(0.3, 901.0, 1.7, 2.5),
        |a| a.aslt_cfg.scale = freq_scale::FreqScale::from_parts(0.3, 901.0, 1.8, 2.5),
        |a| a.aslt_cfg.scale = freq_scale::FreqScale::from_parts(0.3, 901.0, 1.8, 2.6),
        |a| a.interp_mode = InterpolationMode::Akima,
        |a| a.eq_weights = vec![1.1; a.bar_count],
        |a| a.rt_pad_factor = 8,
        |a| a.fft_size = 8192,
    ];
    let base = |a: &mut SpectrumAnalyzer| {
        a.fft_size = 4096;
        a.bar_count = 333;
        a.live_rate = 48000;
        a.sample_rate = 48000;
        a.bar_mapping = BarMappingMode::FlatOverlap;
    };
    let mut prod = realtime_route_tests::rt_analyzer(&base, false);
    let mut old = realtime_route_tests::rt_analyzer(&base, true);
    realtime_route_tests::drive_pair("overlap base", &mut prod, &mut old, 2);
    for (i, edit) in changes.iter().enumerate() {
        for a in [&mut prod, &mut old] {
            edit(a);
            a.rebuild_realtime_fft();
        }
        realtime_route_tests::drive_pair(&format!("overlap edit {i}"), &mut prod, &mut old, 2);
    }
    for mapping in [
        BarMappingMode::Gaussian,
        BarMappingMode::FlatOverlap,
        BarMappingMode::Gaussian,
    ] {
        for a in [&mut prod, &mut old] {
            a.bar_mapping = mapping.clone();
        }
        realtime_route_tests::drive_pair("overlap mapping transition", &mut prod, &mut old, 2);
    }
}

#[test]
#[ignore = "exclusive release-only live analyser screen, not whole-player CPU or GUI timing"]
fn release_live_padding_and_mapping_screen() {
    use serde_json::json;
    use std::hint::black_box;
    assert!(
        !std::hint::black_box(cfg!(debug_assertions)),
        "run performance screen in release"
    );
    let wide = std::env::var_os("MOOSIK_RT_WIDE").is_some();
    let geometries = if wide {
        vec![
            (4096, 44100, 1024),
            (8192, 48000, 2048),
            (16384, 96000, 2048),
        ]
    } else {
        vec![(8192, 48000, 2048)]
    };
    for (size, sr, bars) in geometries {
        let inputs: Vec<_> = (0..4)
            .map(|i| {
                let l = pcm(size, 0.3 + i as f32 * 0.21);
                let r = pcm(size, 1.1 - i as f32 * 0.17);
                let mix: Vec<_> = l.iter().zip(&r).map(|(l, r)| (l + r) * 0.5).collect();
                (mix, l, r)
            })
            .collect();
        for pad in [2, 4, 8] {
            for (name, mapping) in [
                ("flat", BarMappingMode::FlatOverlap),
                ("gaussian", BarMappingMode::Gaussian),
                ("cqt-weighting", BarMappingMode::Cqt),
            ] {
                for joint in [false, true] {
                    // 0 labels an all-identical best case, not an estimate of
                    // real playback's source publication cadence.
                    for repeat in [1, 2, 0] {
                        let set = |a: &mut SpectrumAnalyzer| {
                            a.fft_size = size;
                            a.rt_pad_factor = pad;
                            a.live_rate = sr;
                            a.sample_rate = sr;
                            a.bar_count = bars;
                            a.bar_mapping = mapping.clone();
                            a.aslt_cfg.scale = freq_scale::FreqScale::Blend(0.5);
                        };
                        let mut new = realtime_route_tests::rt_analyzer(&set, false);
                        // Current 22ec591 already deduplicates norms and caches
                        // CQT weighting. Do not time the older frozen wrapper,
                        // which would add an already-removed norms allocation.
                        let mut old = realtime_route_tests::rt_analyzer(&set, false);
                        old.rt_uncached_overlap = !matches!(mapping, BarMappingMode::Cqt);
                        old.rt_disable_input_reuse = true;
                        for a in [&mut new, &mut old] {
                            a.waterfall_enabled = false;
                        }
                        let publish = |a: &mut SpectrumAnalyzer, index: usize| {
                            let mut buffer = a.sample_buf.lock().unwrap();
                            buffer.clear();
                            buffer.extend_from_slice(&inputs[index].0);
                        };
                        let at = |tick: usize| if repeat == 0 { 0 } else { (tick / repeat) % 4 };
                        let run = |a: &mut SpectrumAnalyzer, index: usize| {
                            a.process_realtime(1.0 / 180.0);
                            if joint {
                                a.process_channels(&inputs[index].1, &inputs[index].2, 1.0 / 180.0);
                            }
                            black_box(&a.magnitudes);
                        };
                        for tick in 0..20 {
                            let index = at(tick);
                            for a in [&mut new, &mut old] {
                                publish(a, index);
                                run(a, index);
                            }
                        }
                        assert_eq!(
                            realtime_route_tests::state(&new),
                            realtime_route_tests::state(&old)
                        );
                        assert_eq!(bits(&new.bars_left), bits(&old.bars_left));
                        assert_eq!(bits(&new.bars_right), bits(&old.bars_right));
                        let measure = |a: &mut SpectrumAnalyzer| {
                            publish(a, 3);
                            run(a, 3);
                            a.rt_fft_computations = 0;
                            a.rt_fft_reuses = 0;
                            a.ch_fft_reuses = 0;
                            let mut samples = Vec::with_capacity(64);
                            for tick in 0..64 {
                                let index = at(tick);
                                // Publication is outside the analyser timer. Its
                                // own locked snapshot/copy/compare remains inside.
                                publish(a, index);
                                let start = Instant::now();
                                run(a, index);
                                samples.push(start.elapsed().as_secs_f64() * 1e6);
                            }
                            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
                            let (computed, reused, ch_reused) = (a.rt_fft_computations, a.rt_fft_reuses, a.ch_fft_reuses);
                            samples.sort_by(f64::total_cmp);
                            publish(a, 0);
                            let count = crate::alloc_probe::arm();
                            run(a, 0);
                            let allocations = crate::alloc_probe::disarm(count);
                            (
                                mean,
                                samples[32],
                                samples[60],
                                samples[63],
                                allocations,
                                computed,
                                reused,
                                ch_reused,
                            )
                        };
                        for rep in 0..3 {
                            let (b, c) = if rep % 2 == 0 {
                                (measure(&mut old), measure(&mut new))
                            } else {
                                let c = measure(&mut new);
                                (measure(&mut old), c)
                            };
                            let expected = match repeat {
                                0 => 1,
                                1 => 64,
                                2 => 32,
                                _ => unreachable!(),
                            };
                            assert_eq!((b.5, b.6, b.7), (64, 0, 0));
                            assert_eq!((c.5, c.6), (expected, 64 - expected));
                            assert_eq!(c.7, if joint { 2 * (64 - expected) } else { 0 });
                            println!(
                                "rt_screen {}",
                                json!({"fft":size,"sr":sr,"bars":bars,"pad":pad,"mapping":name,"joint":joint,"repeat":repeat,"rep":rep,
                            "baseline_mean_us":b.0,"baseline_p50_us":b.1,"baseline_p95_us":b.2,"baseline_p99_us":b.3,"baseline_allocs":b.4,"baseline_ffts":b.5,"baseline_reuses":b.6,
                            "candidate_mean_us":c.0,"candidate_p50_us":c.1,"candidate_p95_us":c.2,"candidate_p99_us":c.3,"candidate_allocs":c.4,"candidate_ffts":c.5,"candidate_reuses":c.6,"candidate_channel_reuses":c.7})
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn each_channel_reuses_only_its_own_identical_window() {
    let set = |a: &mut SpectrumAnalyzer| {
        a.fft_size = 1024;
        a.rt_pad_factor = 4;
        a.bar_count = 256;
        a.bar_mapping = BarMappingMode::Cqt;
    };
    let mut new = realtime_route_tests::rt_analyzer(&set, false);
    let mut old = realtime_route_tests::rt_analyzer(&set, false);
    old.rt_disable_input_reuse = true;
    let (a, b) = (pcm(1500, 0.4), pcm(1500, 1.3));
    let mut z = a.clone();
    z[1499] = -0.0;
    let mut zp = a.clone();
    zp[1499] = 0.0;
    // (left, right, transforms the candidate must run)
    let steps: [(&[f32], &[f32], u64); 9] = [
        (&a, &b, 2),
        (&a, &b, 0),   // both held
        (&b, &b, 1),   // left changes; right does not copy left's state
        (&b, &a, 1),   // swapped sides are not reused across sides
        (&b, &a, 0),
        (&z, &a, 1),   // signed zero is a different window
        (&zp, &a, 1),
        (&a[..1400], &a[100..], 1), // only the newest fft_size samples count
        (&a[100..], &a[100..], 1),  // left equals right's window, not its own
    ];
    for (i, (l, r, want)) in steps.iter().enumerate() {
        let before = new.take_channel_ffts();
        assert_eq!(before, 0);
        new.process_channels(l, r, 1.0 / 180.0);
        old.process_channels(l, r, 1.0 / 180.0);
        assert_eq!(new.take_channel_ffts(), *want, "step {i}");
        assert_eq!(old.take_channel_ffts(), 2);
        assert_eq!(bits(&new.bars_left), bits(&old.bars_left), "step {i}");
        assert_eq!(bits(&new.bars_right), bits(&old.bars_right), "step {i}");
        assert_eq!(bits(&new.ch_norms[0]), bits(&old.ch_norms[0]), "step {i}");
        assert_eq!(bits(&new.ch_norms[1]), bits(&old.ch_norms[1]), "step {i}");
    }
    // Same-size window rebuild and padding change both invalidate both sides.
    for a in [&mut new, &mut old] {
        a.window_fn = WindowFn::Blackman;
        a.rebuild_realtime_fft();
    }
    new.process_channels(&a[100..], &a[100..], 1.0 / 180.0);
    old.process_channels(&a[100..], &a[100..], 1.0 / 180.0);
    assert_eq!(new.take_channel_ffts(), 2, "rebuilt window reused an old transform");
    assert_eq!(bits(&new.bars_left), bits(&old.bars_left));
    for a in [&mut new, &mut old] {
        a.rt_pad_factor = 8;
        a.rebuild_realtime_fft();
        a.process_channels(&b, &b, 1.0 / 180.0);
    }
    assert_eq!(new.take_channel_ffts(), 2);
    assert_eq!(bits(&new.bars_right), bits(&old.bars_right));
    // A norm vector of the wrong length (e.g. cleared) forces a transform.
    new.ch_norms[0].clear();
    new.process_channels(&b, &b, 1.0 / 180.0);
    assert_eq!(new.take_channel_ffts(), 1);
}

#[test]
#[ignore = "exclusive release-only probe: cold geometry and mapped-output allocation"]
fn release_cold_geometry_and_output_allocation_probe() {
    use serde_json::json;
    use std::hint::black_box;
    assert!(!black_box(cfg!(debug_assertions)), "run in release");
    for (size, sr, bars) in [(4096, 44100, 1024), (8192, 48000, 2048), (16384, 96000, 2048)] {
        for pad in [2, 8] {
            for (name, mapping) in [
                ("flat", BarMappingMode::FlatOverlap),
                ("gaussian", BarMappingMode::Gaussian),
                ("cqt-weighting", BarMappingMode::Cqt),
            ] {
                let set = |a: &mut SpectrumAnalyzer| {
                    a.fft_size = size;
                    a.rt_pad_factor = pad;
                    a.live_rate = sr;
                    a.sample_rate = sr;
                    a.bar_count = bars;
                    a.bar_mapping = mapping.clone();
                    a.aslt_cfg.scale = freq_scale::FreqScale::Blend(0.5);
                };
                let mut a = realtime_route_tests::rt_analyzer(&set, false);
                let padded = a.realtime_padded_size();
                let mut cold = Vec::new();
                for _ in 0..15 {
                    a.overlap_plan = None;
                    a.cqt_plan = None;
                    let t = Instant::now();
                    if matches!(mapping, BarMappingMode::Cqt) {
                        a.ensure_cqt_plan(sr, bars, padded);
                    } else {
                        a.ensure_overlap_plan(sr, bars, padded);
                    }
                    cold.push(t.elapsed().as_secs_f64() * 1e6);
                }
                cold.sort_by(f64::total_cmp);
                // The edge evaluations alone: the most an n+1 edge share could save
                // is half of these (2n -> n+1 inversions).
                let scale = a.analysis_scale();
                let hi = a.live_mapping_max_freq(sr);
                let mut edges = Vec::new();
                for _ in 0..15 {
                    let t = Instant::now();
                    for bar in 0..bars {
                        black_box(scale.freq_at(bar as f32 / bars as f32, a.min_freq, hi));
                        black_box(scale.freq_at((bar + 1) as f32 / bars as f32, a.min_freq, hi));
                    }
                    edges.push(t.elapsed().as_secs_f64() * 1e6);
                }
                edges.sort_by(f64::total_cmp);
                println!("cold_geometry {}", json!({"fft":size,"sr":sr,"bars":bars,"pad":pad,"mapping":name,
                    "cold_median_us":cold[7],"cold_min_us":cold[0],"two_n_edges_median_us":edges[7]}));
            }
        }
        // Mapped-output allocation: a fresh n-element collect versus refilling a kept one.
        let mut fresh = Vec::new();
        let mut kept_t = Vec::new();
        let mut kept = vec![0.0f32; bars];
        for _ in 0..15 {
            let t = Instant::now();
            for i in 0..1000 {
                let v: Vec<f32> = (0..bars).map(|b| (b + i) as f32).collect();
                black_box(&v);
            }
            fresh.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            for i in 0..1000 {
                kept.clear();
                kept.extend((0..bars).map(|b| (b + i) as f32));
                black_box(&kept);
            }
            kept_t.push(t.elapsed().as_secs_f64() * 1e3);
        }
        fresh.sort_by(f64::total_cmp);
        kept_t.sort_by(f64::total_cmp);
        println!("output_alloc {}", json!({"bars":bars,"fresh_collect_us":fresh[7],"kept_refill_us":kept_t[7]}));
    }
}

#[test]
#[ignore = "exclusive release-only paired probe: per-side reuse overhead on fresh input"]
fn release_channel_reuse_paired_probe() {
    use serde_json::json;
    use std::hint::black_box;
    assert!(!black_box(cfg!(debug_assertions)), "run in release");
    for (size, sr, bars, pad) in [(4096, 44100, 1024, 2), (8192, 48000, 2048, 2), (8192, 48000, 2048, 8), (16384, 96000, 2048, 8)] {
        for repeat in [1, 2] {
            let set = |a: &mut SpectrumAnalyzer| {
                a.fft_size = size;
                a.rt_pad_factor = pad;
                a.live_rate = sr;
                a.sample_rate = sr;
                a.bar_count = bars;
                a.bar_mapping = BarMappingMode::Cqt;
                a.aslt_cfg.scale = freq_scale::FreqScale::Blend(0.5);
            };
            let inputs: Vec<_> = (0..4)
                .map(|i| {
                    let l = pcm(size, 0.3 + i as f32 * 0.21);
                    let r = pcm(size, 1.1 - i as f32 * 0.17);
                    let mix: Vec<_> = l.iter().zip(&r).map(|(l, r)| (l + r) * 0.5).collect();
                    (mix, l, r)
                })
                .collect();
            let mut new = realtime_route_tests::rt_analyzer(&set, false);
            let mut old = realtime_route_tests::rt_analyzer(&set, false);
            old.rt_disable_channel_reuse = true;
            let run = |a: &mut SpectrumAnalyzer, index: usize| {
                {
                    let mut buffer = a.sample_buf.lock().unwrap();
                    buffer.clear();
                    buffer.extend_from_slice(&inputs[index].0);
                }
                let start = Instant::now();
                a.process_realtime(1.0 / 180.0);
                a.process_channels(&inputs[index].1, &inputs[index].2, 1.0 / 180.0);
                black_box(&a.bars_right);
                start.elapsed().as_secs_f64() * 1e6
            };
            for tick in 0..20 {
                run(&mut new, (tick / repeat) % 4);
                run(&mut old, (tick / repeat) % 4);
            }
            assert_eq!(bits(&new.bars_left), bits(&old.bars_left));
            assert_eq!(bits(&new.bars_right), bits(&old.bars_right));
            let measure = |a: &mut SpectrumAnalyzer| (0..64).map(|t| run(a, (t / repeat) % 4)).sum::<f64>() / 64.0;
            let mut ratios = Vec::new();
            let (mut bs, mut cs) = (Vec::new(), Vec::new());
            for round in 0..15 {
                let (b, c) = if round % 2 == 0 {
                    (measure(&mut old), measure(&mut new))
                } else {
                    let c = measure(&mut new);
                    (measure(&mut old), c)
                };
                bs.push(b);
                cs.push(c);
                ratios.push(b / c);
            }
            assert_eq!(bits(&new.bars_right), bits(&old.bars_right));
            for v in [&mut bs, &mut cs, &mut ratios] {
                v.sort_by(f64::total_cmp);
            }
            println!("lr_paired {}", json!({"fft":size,"bars":bars,"pad":pad,"repeat":repeat,
                "no_lr_reuse_median_us":bs[7],"lr_reuse_median_us":cs[7],
                "ratio_median":ratios[7],"ratio_min":ratios[0],"ratio_max":ratios[14],"ratio_q1":ratios[3],"ratio_q3":ratios[11]}));
        }
    }
}

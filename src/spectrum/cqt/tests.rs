use super::*;
use std::cell::Cell;

fn code(value: f32) -> u16 {
    // This is the production mapping, not a test's approximation to it.
    crate::spectrum::aslt_code(value)
}

#[test]
fn coarse_phase_diagnostics_change_neither_coefficients_codes_work_nor_progress() {
    use std::cell::RefCell;
    // Low/mid groups contain several equal-size kernels; the treble group is
    // below 512 at 12 BPO and checks the combined serial region too. At 48
    // BPO even a Nyquist-adjacent filter needs N>=512, so it cannot prove this.
    let frequencies = [180.0, 190.0, 200.0, 210.0, 230.0, 250.0,
        1000.0, 1100.0, 1200.0, 1300.0, 10000.0, 11000.0];
    for (sr, workers, window) in [(44100, 1, Window::Hann), (48000, 3, Window::Blackman)] {
        let left = signal(sr, 8193, 4);
        let right = signal(sr, 8193, 2);
        let cfg = Config { bins_per_octave: 12, frame_rate: 179.5, window, bass_limit_s: None };
        let frame_len = frame_count(left.len(), sr, cfg.frame_rate).unwrap();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap();
        let replay = |enabled| pool.install(|| {
            let bits = RefCell::new(vec![None; frame_len * frequencies.len()]);
            let progress = RefCell::new(Vec::new());
            let observer = |frame: usize, bin: usize, left: Complex<f64>, right: Option<Complex<f64>>| {
                let right = right.expect("joint coefficient");
                let value = [left.re.to_bits(), left.im.to_bits(), right.re.to_bits(), right.im.to_bits()];
                assert!(bits.borrow_mut()[frame * frequencies.len() + bin].replace(value).is_none());
            };
            let output = analyze_inner_profiled(
                Input::Joint { left: &left, right: &right }, sr, &frequencies, &cfg,
                Limits::default(), &|| true, &|value| progress.borrow_mut().push(value),
                &code, Some(&observer), false, enabled,
            ).unwrap();
            assert!(bits.borrow().iter().all(Option::is_some));
            (output, bits.into_inner(), progress.into_inner())
        });
        let (off, off_bits, off_progress) = replay(false);
        let (on, on_bits, on_progress) = replay(true);
        assert_eq!(on.codes, off.codes);
        assert_eq!(on_bits, off_bits);
        assert_eq!(on_progress, off_progress);
        assert!(on_progress.windows(2).all(|values| values[0] <= values[1]));
        let mut on_work = on.work;
        let phases = on_work.phases;
        on_work.phases = PhaseTimings::default();
        assert_eq!(on_work, off.work);
        assert_eq!(off.work.phases, PhaseTimings::default());
        assert!(phases.enabled && phases.total_wall_ns > 0);
        assert_eq!(phases.kernel_batches, on.work.batches);
        assert_eq!(phases.source_blocks * 2, on.work.input_ffts);
        assert_eq!(phases.source_blocks, phases.parallel_blocks + phases.serial_blocks);
        let named_wall = phases.setup_wall_ns + phases.planning_wall_ns
            + phases.allocation_wall_ns + phases.kernel_prepare_fft_wall_ns
            + phases.source_fill_fft_wall_ns + phases.parallel_inverse_region_wall_ns
            + phases.parallel_fold_progress_wall_ns + phases.serial_inverse_fold_progress_wall_ns;
        assert_eq!(phases.total_wall_ns, named_wall + phases.unclassified_wall_ns);
        if workers > 1 {
            assert!(phases.parallel_blocks > 0 && phases.parallel_tasks >= phases.parallel_blocks);
            assert!(phases.max_parallel_tasks_per_block <= workers as u64);
            assert!(phases.parallel_inverse_worker_ns > 0);
        } else {
            assert_eq!(phases.parallel_blocks, 0);
            assert_eq!(phases.parallel_tasks, 0);
            assert_eq!(phases.parallel_inverse_worker_ns, 0);
        }
        assert!(phases.serial_blocks > 0);
    }
}

fn config(window: Window) -> Config {
    Config {
        bins_per_octave: 24,
        window,
        frame_rate: 180.0,
        bass_limit_s: None,
    }
}

fn signal(sr: u32, length: usize, kind: usize) -> Vec<f32> {
    let mut seed = 0x155_u64;
    (0..length)
        .map(|i| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let noise = ((seed >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0) * 0.13;
            let t = i as f64 / f64::from(sr);
            (match kind {
                0 => 0.0,
                1 => {
                    if i == length / 2 {
                        1.0
                    } else {
                        0.0
                    }
                }
                2 => 0.71 * (TAU * 440.0 * t + 0.37).cos() + 0.17 * (TAU * 1730.0 * t + 1.3).sin(),
                3 => noise,
                4 => 0.6 * (TAU * (80.0 * t + 2700.0 * t * t)).sin() + noise,
                _ => {
                    if i > length / 3 && i < length * 2 / 3 {
                        0.8 * (TAU * 440.0 * t).cos()
                    } else {
                        0.0
                    }
                }
            }) as f32
        })
        .collect()
}

fn add_compensated(sum: &mut f64, correction: &mut f64, value: f64) {
    let next = *sum + value;
    *correction += if sum.abs() >= value.abs() {
        (*sum - next) + value
    } else {
        (value - next) + *sum
    };
    *sum = next;
}

/// Independent direct reference: support is re-derived with exp_m1, window
/// uses its conventional [0,1] definition, and both real/imag sums are
/// compensated. It calls neither the optimized kernel nor its window builder.
fn reference(source: &[f32], centre: usize, sr: u32, f: f64, config: Config) -> Complex<f64> {
    let q = 1.0 / (std::f64::consts::LN_2 / f64::from(config.bins_per_octave)).exp_m1();
    // Responsive: the VQT bandwidth f/Q + 1/t in Hz, half its reciprocal in samples.
    let half = match config.bass_limit_s {
        None => (q * f64::from(sr) / (2.0 * f)).ceil() as usize,
        Some(t) => (f64::from(sr) / (2.0 * (f / q + 1.0 / f64::from(t)))).ceil() as usize,
    };
    let (mut re, mut rc, mut im, mut ic, mut sum, mut sc) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..=2 * half {
        let u = i as f64 / (2 * half) as f64;
        let w = match config.window {
            Window::Hann => (PI * u).sin().powi(2),
            Window::Hamming => 0.54 - 0.46 * (TAU * u).cos(),
            Window::Blackman => 0.42 - 0.5 * (TAU * u).cos() + 0.08 * (2.0 * TAU * u).cos(),
            Window::BlackmanHarris => {
                0.35875 - 0.48829 * (TAU * u).cos() + 0.14128 * (2.0 * TAU * u).cos()
                    - 0.01168 * (3.0 * TAU * u).cos()
            }
        };
        add_compensated(&mut sum, &mut sc, w);
        let offset = i as i64 - half as i64;
        let at = centre as i64 + offset;
        let x = if at < 0 {
            0.0
        } else {
            source.get(at as usize).copied().unwrap_or(0.0)
        };
        let angle = TAU * f * offset as f64 / f64::from(sr);
        add_compensated(&mut re, &mut rc, f64::from(x) * w * angle.cos());
        add_compensated(&mut im, &mut ic, -f64::from(x) * w * angle.sin());
    }
    Complex::new(re + rc, im + ic) * (2.0 / (sum + sc))
}

fn analyze(source: &[f32], sr: u32, frequencies: &[f64], config: Config) -> Output {
    analyze_input_codes(
        Input::Mono(source),
        sr,
        frequencies,
        &config,
        Limits::default(),
        &|| true,
        &|_| {},
        &code,
    )
    .unwrap()
}

#[test]
fn dense_codes_match_independent_f64_reference_at_three_sample_rates_all_windows() {
    for window in [
        Window::Hann,
        Window::Hamming,
        Window::Blackman,
        Window::BlackmanHarris,
    ] {
        let config = config(window);
        for sr in [44100, 48000, 96000] {
            for kind in [0, 1, 2, 3, 4, 5] {
                let source = signal(sr, 2049, kind);
                let frequencies = [440.0, 1730.0, 4000.0];
                let result = analyze(&source, sr, &frequencies, config);
                for frame in 0..result.frames {
                    // Independent clock, not frame_center.
                    let centre = (frame as f64 * f64::from(sr) / f64::from(config.frame_rate))
                        .floor() as usize;
                    for (bar, &f) in frequencies.iter().enumerate() {
                        let expected =
                            code(reference(&source, centre, sr, f, config).norm() as f32);
                        assert_eq!(
                            result.codes[0][frame * frequencies.len() + bar],
                            expected,
                            "window={window:?} sr={sr} kind={kind} frame={frame} f={f}"
                        );
                    }
                }
                assert!(result.codes[1].is_empty());
                assert!(result.codes[2].is_empty());
            }
        }
    }
}

/// The Responsive variable-Q transform runs through the same engine with its
/// own shorter supports; its codes equal the independent direct reference,
/// and its supports really differ from Full CQT's where `Q/f` is long.
#[test]
fn responsive_codes_match_independent_f64_reference_and_shorten_bass_support() {
    let sr = 48_000;
    let frequencies = [55.0, 440.0, 1730.0, 4000.0];
    for window in [Window::Hann, Window::Hamming, Window::Blackman, Window::BlackmanHarris] {
        for t in [0.05f32, DEFAULT_RESPONSIVE_LIMIT_S, 0.5] {
            let full = config(window);
            let config = Config { bass_limit_s: Some(t), ..full };
            for &f in &frequencies {
                let (s, q) = (config.support_seconds(f), full.support_seconds(f));
                assert!(s < q && s < f64::from(t), "support {s} not below Q/f {q} and t {t}");
            }
            assert!(full.support_seconds(55.0) - config.support_seconds(55.0) > 0.1);
            for kind in [1, 2, 4, 5] {
                let source = signal(sr, 2049, kind);
                let result = analyze(&source, sr, &frequencies, config);
                let full_result = analyze(&source, sr, &frequencies, full);
                for frame in 0..result.frames {
                    let centre = (frame as f64 * f64::from(sr) / f64::from(config.frame_rate)).floor() as usize;
                    for (bar, &f) in frequencies.iter().enumerate() {
                        let expected = code(reference(&source, centre, sr, f, config).norm() as f32);
                        assert_eq!(
                            result.codes[0][frame * frequencies.len() + bar],
                            expected,
                            "window={window:?} t={t} kind={kind} frame={frame} f={f}"
                        );
                    }
                }
                if kind == 2 {
                    assert_ne!(result.codes[0], full_result.codes[0], "Responsive equalled Full CQT");
                }
            }
        }
    }
    let bad = |t: f32| Config { bass_limit_s: Some(t), ..Config::default() }.validate().is_err();
    assert!(bad(0.049) && bad(10.01) && bad(f32::NAN) && bad(f32::INFINITY));
    assert!(!bad(0.05) && !bad(10.0));
}

#[test]
fn actual_production_complex_coefficients_match_independent_direct_reference() {
    let sr = 44100;
    let left = signal(sr, 8001, 4);
    let right = signal(sr, 8001, 2);
    let frequencies = [55.0, 440.0, 1730.0, 4000.0];
    for window in [
        Window::Hann,
        Window::Hamming,
        Window::Blackman,
        Window::BlackmanHarris,
    ] {
        let cfg = Config {
            frame_rate: 179.5,
            ..config(window)
        };
        let seen = Cell::new(0);
        let observer = |frame: usize, bar: usize, l: Complex<f64>, r: Option<Complex<f64>>| {
            let centre = (frame as f64 * f64::from(sr) / 179.5).floor() as usize;
            let expected_l = reference(&left, centre, sr, frequencies[bar], cfg);
            let expected_r = reference(&right, centre, sr, frequencies[bar], cfg);
            assert!(
                (l - expected_l).norm() < 3e-12,
                "left window={window:?} frame={frame} bar={bar} got={l:?} expected={expected_l:?}"
            );
            assert!((r.unwrap() - expected_r).norm() < 3e-12);
            seen.set(seen.get() + 1);
        };
        let out = analyze_inner(
            Input::Joint {
                left: &left,
                right: &right,
            },
            sr,
            &frequencies,
            &cfg,
            Limits::default(),
            &|| true,
            &|_| {},
            &code,
            Some(&observer),
            false,
        )
        .unwrap();
        assert_eq!(seen.get(), out.frames * out.bars);
    }
}

#[test]
fn tone_gain_and_impulse_edges_use_full_window_normalization() {
    let sr = 48000;
    let f = 1000.0;
    for window in [
        Window::Hann,
        Window::Hamming,
        Window::Blackman,
        Window::BlackmanHarris,
    ] {
        let config = Config {
            bins_per_octave: 48,
            frame_rate: 180.0,
            window,
            bass_limit_s: None,
        };
        let source: Vec<f32> = (0..9600)
            .map(|i| (0.7 * (TAU * f * i as f64 / f64::from(sr) + 0.3).cos()) as f32)
            .collect();
        let result = analyze(&source, sr, &[f], config);
        let interior = 18;
        let reference_gain = reference(&source, 4800, sr, f, config).norm();
        // Hamming has a nonzero endpoint; finite negative-frequency leakage
        // is measured rather than assumed identically zero.
        assert!((reference_gain - 0.7).abs() < 0.001);
        assert_eq!(result.codes[0][interior], code(reference_gain as f32));
        assert!(result.codes[0][0] < result.codes[0][interior]);

        let mut impulse = vec![0.0; 9600];
        impulse[0] = 1.0;
        let result = analyze(&impulse, sr, &[f], config);
        for frame in [0, 1, 2, result.frames - 1] {
            let centre = frame_center(frame, sr, config.frame_rate);
            assert_eq!(
                result.codes[0][frame],
                code(reference(&impulse, centre, sr, f, config).norm() as f32)
            );
        }
        assert_eq!(*result.codes[0].last().unwrap(), code(0.0));
    }
}

#[test]
fn joint_mix_is_complex_mean_and_uses_two_not_three_convolutions() {
    let sr = 44100;
    let source = signal(sr, 8001, 2);
    let opposite: Vec<_> = source.iter().map(|v| -*v).collect();
    let frequencies = [220.0, 440.0, 1730.0];
    let cfg = config(Window::Blackman);
    let mono = analyze(&source, sr, &frequencies, cfg);
    for right in [&source, &opposite] {
        let result = analyze_input_codes(
            Input::Joint {
                left: &source,
                right,
            },
            sr,
            &frequencies,
            &cfg,
            Limits::default(),
            &|| true,
            &|_| {},
            &code,
        )
        .unwrap();
        assert_eq!(result.codes[1], mono.codes[0]);
        assert_eq!(result.codes[2], mono.codes[0]);
        if std::ptr::eq(right, &source) {
            assert_eq!(result.codes[0], mono.codes[0]);
        } else {
            assert!(result.codes[0].iter().all(|&v| v == code(0.0)));
            assert!(result.codes[1].iter().any(|&v| v > code(0.0)));
        }
        assert_eq!(result.work.kernel_ffts, mono.work.kernel_ffts);
        assert_eq!(result.work.input_ffts, 2 * mono.work.input_ffts);
        assert_eq!(result.work.inverse_ffts, 2 * mono.work.inverse_ffts);
        assert_eq!(
            result.work.column_convolutions,
            2 * mono.work.column_convolutions
        );
    }
}

#[test]
fn arbitrary_stereo_matches_independent_complex_oracle_on_fractional_hops() {
    let sr = 44100;
    let left = signal(sr, 3301, 4);
    let right = signal(sr, 3301, 2);
    let cfg = Config {
        frame_rate: 179.5,
        ..config(Window::Hamming)
    };
    let frequencies = [180.0, 440.0, 4000.0];
    let out = analyze_input_codes(
        Input::Joint {
            left: &left,
            right: &right,
        },
        sr,
        &frequencies,
        &cfg,
        Limits::default(),
        &|| true,
        &|_| {},
        &code,
    )
    .unwrap();
    for frame in 0..out.frames {
        let centre = (frame as f64 * f64::from(sr) / 179.5).floor() as usize;
        for (bar, &f) in frequencies.iter().enumerate() {
            let l = reference(&left, centre, sr, f, cfg);
            let r = reference(&right, centre, sr, f, cfg);
            let at = frame * frequencies.len() + bar;
            assert_eq!(out.codes[0][at], code(((l + r) * 0.5).norm() as f32));
            assert_eq!(out.codes[1][at], code(l.norm() as f32));
            assert_eq!(out.codes[2][at], code(r.norm() as f32));
        }
    }
}

#[test]
fn workspace_changes_batches_not_results_and_resource_refusal_never_caps() {
    let source = signal(48000, 4000, 3);
    let frequencies: Vec<_> = (0..20).map(|i| 440.0 + i as f64).collect();
    let cfg = config(Window::Blackman);
    // Admit exactly one spectrum alongside the real plan's required mono
    // source/product/scratch. A hand-picked 448 KiB assumed zero scratch;
    // rustfft's actual SIMD plan does not make that promise.
    let groups = layout(&frequencies, 48000, &cfg, Limits::default()).unwrap();
    assert_eq!(groups.len(), 1);
    let n = *groups.keys().next().unwrap();
    let mut planner = FftPlanner::<f64>::new();
    let scratch = planner
        .plan_fft_forward(n)
        .get_inplace_scratch_len()
        .max(planner.plan_fft_inverse(n).get_inplace_scratch_len());
    let small = Limits {
        workspace_bytes: (3 * n + scratch) * size_of::<Complex<f64>>(),
        ..Limits::default()
    };
    let a = analyze_input_codes(
        Input::Mono(&source),
        48000,
        &frequencies,
        &cfg,
        small,
        &|| true,
        &|_| {},
        &code,
    )
    .unwrap();
    let b = analyze(&source, 48000, &frequencies, cfg);
    assert_eq!(a.codes, b.codes);
    assert!(a.work.batches > b.work.batches);
    assert!(a.work.input_ffts > b.work.input_ffts);
    assert!(a.work.peak_workspace_bytes <= small.workspace_bytes);
    assert_eq!(a.work.output_bytes, a.frames * a.bars * 2);
    assert_eq!(
        a.work.inverses_pruned, 0,
        "optional workspace must not displace the one admitted kernel"
    );
    assert!(a.work.inverses_full > 0);
    let too_small = Limits {
        workspace_bytes: 1024,
        ..Limits::default()
    };
    assert!(matches!(
        analyze_input_codes(
            Input::Mono(&source),
            48000,
            &frequencies,
            &cfg,
            too_small,
            &|| true,
            &|_| {},
            &code
        ),
        Err(Error::Resource(_))
    ));
    let fft_limit = Limits {
        max_fft_len: 1024,
        ..Limits::default()
    };
    let err = analyze_input_codes(
        Input::Mono(&source),
        48000,
        &[20.0],
        &cfg,
        fft_limit,
        &|| true,
        &|_| {},
        &code,
    )
    .unwrap_err();
    assert!(err.to_string().contains("no filter was shortened"));
    let output_limit = Limits {
        output_bytes: 3,
        ..Limits::default()
    };
    assert!(matches!(
        analyze_input_codes(
            Input::Mono(&source),
            48000,
            &frequencies,
            &cfg,
            output_limit,
            &|| true,
            &|_| {},
            &code
        ),
        Err(Error::Resource(_))
    ));
}

#[test]
fn cancellation_at_start_during_kernel_and_after_progress_returns_no_partial_plane() {
    let source = signal(48000, 14000, 3);
    let cfg = config(Window::Hann);
    let calls = std::sync::atomic::AtomicUsize::new(0);
    for allowed in [0, 4, 20] {
        calls.store(0, std::sync::atomic::Ordering::Relaxed);
        let err = analyze_input_codes(
            Input::Mono(&source),
            48000,
            &[20.0, 55.0],
            &cfg,
            Limits::default(),
            &|| {
                let call = calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                call < allowed
            },
            &|_| {},
            &code,
        )
        .unwrap_err();
        assert_eq!(err, Error::Aborted);
    }
    let keep_going = std::sync::atomic::AtomicBool::new(true);
    let positive_progress = Cell::new(false);
    let err = analyze_input_codes(
        Input::Mono(&source),
        48000,
        &[440.0, 450.0],
        &cfg,
        Limits::default(),
        &|| keep_going.load(std::sync::atomic::Ordering::Relaxed),
        &|p| {
            if p > 0.0 {
                positive_progress.set(true);
                keep_going.store(false, std::sync::atomic::Ordering::Relaxed);
            }
        },
        &code,
    )
    .unwrap_err();
    assert_eq!(err, Error::Aborted);
    assert!(positive_progress.get());
}

#[test]
fn configuration_clock_and_invalid_inputs_are_explicit() {
    assert_eq!(frame_count(44100, 44100, 180.0).unwrap(), 180);
    assert_eq!(frame_count(44101, 44100, 180.0).unwrap(), 181);
    assert_eq!(frame_center(180, 44100, 180.0), 44100);
    assert_eq!(frame_center(180, 48000, 180.0), 48000);
    assert_eq!(frame_center(1795, 44100, 179.5), 441000);
    let cfg = Config::default();
    assert!(cfg.validate().is_ok());
    assert!((cfg.support_seconds(20.0) / cfg.support_seconds(200.0) - 10.0).abs() < 1e-12);
    let missing: Config = serde_json::from_str("{}").unwrap();
    assert_eq!(missing, cfg);
    for bad in [
        Config {
            bins_per_octave: 0,
            ..cfg
        },
        Config {
            frame_rate: f32::NAN,
            ..cfg
        },
        Config {
            frame_rate: 0.0,
            ..cfg
        },
    ] {
        assert!(bad.validate().is_err());
    }
    let source = [0.0; 10];
    for bad in [
        vec![],
        vec![0.0],
        vec![24000.0],
        vec![440.0, 440.0],
        vec![f64::NAN],
    ] {
        assert!(
            analyze_input_codes(
                Input::Mono(&source),
                48000,
                &bad,
                &cfg,
                Limits::default(),
                &|| true,
                &|_| {},
                &code
            )
            .is_err()
        );
    }
    assert!(matches!(
        analyze_input_codes(
            Input::Joint {
                left: &source,
                right: &source[..9]
            },
            48000,
            &[440.0],
            &cfg,
            Limits::default(),
            &|| true,
            &|_| {},
            &code
        ),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(
        analyze_input_codes(
            Input::Mono(&[f32::NAN]),
            48000,
            &[440.0],
            &cfg,
            Limits::default(),
            &|| true,
            &|_| {},
            &code
        ),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn progress_is_monotone_and_finished_planes_are_frame_major() {
    let source = signal(44100, 11025, 4);
    let progress = std::cell::RefCell::new(Vec::new());
    let out = analyze_input_codes(
        Input::Mono(&source),
        44100,
        &[100.0, 440.0, 4000.0],
        &config(Window::Blackman),
        Limits::default(),
        &|| true,
        &|p| progress.borrow_mut().push(p),
        &code,
    )
    .unwrap();
    let progress = progress.borrow();
    assert_eq!(progress[0], 0.0);
    assert_eq!(*progress.last().unwrap(), 1.0);
    assert!(progress.windows(2).all(|pair| pair[0] <= pair[1]));
    assert_eq!(out.codes[0].len(), out.frames * out.bars);
    assert_eq!(out.work.column_convolutions, 3);
}

fn replay_with_complex_bits(
    input: Input<'_>,
    sr: u32,
    frequencies: &[f64],
    cfg: Config,
    full: bool,
) -> (Output, Vec<Option<[u64; 4]>>) {
    let samples = match input {
        Input::Mono(x) => x.len(),
        Input::Joint { left, .. } => left.len(),
    };
    let frames = frame_count(samples, sr, cfg.frame_rate).unwrap();
    let bits = std::cell::RefCell::new(vec![None; frames * frequencies.len()]);
    let observer = |frame: usize, bar: usize, l: Complex<f64>, r: Option<Complex<f64>>| {
        let r = r.unwrap_or_default();
        let mut bits = bits.borrow_mut();
        let slot = &mut bits[frame * frequencies.len() + bar];
        assert!(
            slot.is_none(),
            "each consumed coefficient must be announced once"
        );
        *slot = Some([
            l.re.to_bits(),
            l.im.to_bits(),
            r.re.to_bits(),
            r.im.to_bits(),
        ]);
    };
    let out = analyze_inner(
        input,
        sr,
        frequencies,
        &cfg,
        Limits::default(),
        &|| true,
        &|_| {},
        &code,
        Some(&observer),
        full,
    )
    .unwrap();
    let bits = bits.into_inner();
    assert!(bits.iter().all(Option::is_some));
    (out, bits)
}

#[test]
fn genuine_low_n_parallel_groups_match_the_retained_full_complex_bits_and_codes() {
    let sr = 48_000;
    let cfg = Config { bins_per_octave: 12, window: Window::Blackman, frame_rate: 179.5, bass_limit_s: None };
    let left = signal(sr, 8193, 4);
    let right = signal(sr, 8193, 2);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap();
    for n in [512, 1024, 2048] {
        let nominal_half = n * 3 / 16;
        let base = cfg.support_q() * f64::from(sr) / (2.0 * nominal_half as f64);
        for bins in [7, 13] {
            let frequencies: Vec<_> = (0..bins).map(|i| base * (1.0 + i as f64 * 0.002)).collect();
            let groups = layout(&frequencies, sr, &cfg, Limits::default()).unwrap();
            assert_eq!(groups.len(), 1);
            assert_eq!(*groups.keys().next().unwrap(), n, "fixture did not exercise its named low N");
            for joint in [false, true] {
                let input = if joint { Input::Joint { left: &left, right: &right } } else { Input::Mono(&left) };
                let (reference, reference_bits) = replay_with_complex_bits(input, sr, &frequencies, cfg, true);
                let (parallel, parallel_bits) = pool.install(|| replay_with_complex_bits(input, sr, &frequencies, cfg, false));
                assert_eq!(parallel_bits, reference_bits, "N={n} bins={bins} joint={joint}");
                assert_eq!(parallel.codes, reference.codes, "N={n} bins={bins} joint={joint}");
                assert_eq!(parallel.work.parallel_groups, 1, "low-N admission was not actually exercised");
                assert_eq!(parallel.work.max_inverse_workers, 6);
                assert_eq!(parallel.work.kernel_ffts, reference.work.kernel_ffts);
                assert_eq!(parallel.work.input_ffts, reference.work.input_ffts);
                assert_eq!(parallel.work.inverse_ffts, reference.work.inverse_ffts);
                assert_eq!(parallel.work.column_convolutions, reference.work.column_convolutions);
                assert!(parallel.work.peak_workspace_bytes <= Limits::default().workspace_bytes);
                assert_eq!(reference.work.inverses_full, reference.work.inverse_ffts);
                assert_eq!(parallel.work.inverses_pruned + parallel.work.inverses_full, parallel.work.inverse_ffts);
            }
        }
    }
}

#[test]
fn requested_inverse_matches_frozen_full_complex_bits_and_codes_across_the_corpus() {
    // The reference switch enters the retained f9a6999 product+whole-inverse
    // body, with the same plan/kernel/block definition. This is an arithmetic
    // route reference, independent of the optimized pruning/gather API, not an
    // independent definition of CQT (the direct f64 tests serve that purpose).
    let frequencies = [55.0, 57.0, 440.0, 451.0, 4000.0, 16000.0];
    let mut pruned_calls = 0;
    for sr in [44100, 48000, 96000] {
        let mut left = signal(sr, 4097, 4);
        left[0] = -0.0;
        left[2048] = 1.0;
        let right = signal(sr, 4097, 2);
        for window in [
            Window::Hann,
            Window::Hamming,
            Window::Blackman,
            Window::BlackmanHarris,
        ] {
            for fps in [179.5, 180.0] {
                let cfg = Config {
                    frame_rate: fps,
                    ..config(window)
                };
                for joint in [false, true] {
                    let input = if joint {
                        Input::Joint {
                            left: &left,
                            right: &right,
                        }
                    } else {
                        Input::Mono(&left)
                    };
                    let (old, old_bits) =
                        replay_with_complex_bits(input, sr, &frequencies, cfg, true);
                    let (new, new_bits) =
                        replay_with_complex_bits(input, sr, &frequencies, cfg, false);
                    assert_eq!(
                        new_bits, old_bits,
                        "complex bits changed sr={sr} window={window:?} fps={fps} joint={joint}"
                    );
                    assert_eq!(new.codes, old.codes);
                    assert_eq!(new.frames, old.frames);
                    assert_eq!(new.work.kernel_ffts, old.work.kernel_ffts);
                    assert_eq!(new.work.column_convolutions, old.work.column_convolutions);
                    assert_eq!(
                        new.work.inverses_pruned + new.work.inverses_full,
                        new.work.inverse_ffts
                    );
                    assert_eq!(old.work.inverses_full, old.work.inverse_ffts);
                    assert_eq!(old.work.inverses_pruned, 0);
                    assert!(new.work.peak_workspace_bytes <= Limits::default().workspace_bytes);
                    pruned_calls += new.work.inverses_pruned;
                }
            }
        }
    }
    let mut planner = FftPlanner::<f64>::new();
    if planner.plan_fft_inverse(16384).prunes() {
        assert!(
            pruned_calls > 0,
            "the reference comparison must actually execute pruning on this host"
        );
    }
}

#[test]
fn pruning_admission_is_bounded_and_dense_duplicate_clocks_keep_the_full_route() {
    assert!(requested_cap(usize::MAX / 4, 1024, usize::MAX / 4, 1, 360.0) <= 256);
    let mut planner = FftPlanner::<f64>::new();
    let inv = planner.plan_fft_inverse(2048);
    let mut request = Requested::new(&*inv, 4, 1).unwrap();
    assert!(
        !request.prepare(0, 5, 48000, 180.0, 0, 0),
        "dense request exceeds the sparse cap"
    );
    assert!(
        !request.prepare(0, 4, 1, 360.0, 0, 0),
        "duplicate indices cannot enter the pruning API"
    );
    assert!(request.prepare(0, 4, 48000, 180.0, 0, 0));

    // Tiny sample clocks deliberately force repeated/dense timestamps through
    // the complete production route. They are valid clocks, not a reason to
    // panic inside rustfft's strictly-increasing contract.
    let source = [1.0, -0.0, -0.5, 0.0];
    for sr in [1, 12] {
        let cfg = Config {
            frame_rate: 360.0,
            ..config(Window::Hann)
        };
        let f = [f64::from(sr) * 0.2];
        let (old, old_bits) = replay_with_complex_bits(Input::Mono(&source), sr, &f, cfg, true);
        let (new, new_bits) = replay_with_complex_bits(Input::Mono(&source), sr, &f, cfg, false);
        assert_eq!(new.codes, old.codes);
        assert_eq!(new_bits, old_bits);
        assert_eq!(new.work.inverses_pruned, 0);
        assert!(new.work.inverses_full > 0);
    }
}

/// Bounded manual release screen, not a whole-app throughput benchmark.
/// Includes layout, FFT planning, kernel preparation, convolution, code mapping
/// and output allocation. Excludes synthetic signal creation and UI/cache I/O.
/// Run exclusively, e.g. cargo test --release ...::bounded_release_probe
/// -- --ignored --nocapture --test-threads=1. One sample per configuration is
/// descriptive observations only, not a confidence interval or speedup claim.
#[test]
#[ignore = "manual bounded performance screen; run exclusively in release mode"]
fn bounded_release_probe() {
    fn digest(codes: &[u16]) -> u64 {
        // Deterministic FNV-1a identity aid, NOT a cryptographic checksum.
        codes
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .fold(0xcbf29ce484222325, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
            })
    }
    let bars = 1024;
    let frequencies: Vec<f64> = (0..bars)
        .map(|i| 20.0 * 1000.0_f64.powf((i as f64 + 0.5) / bars as f64))
        .collect();
    let cfg = Config::default();
    println!(
        "cqt_probe,mode,sample_rate,seconds,bars,bpo,fps,wall_seconds,frames,kernel_ffts,input_ffts,inverse_ffts,batches,peak_explicit_workspace_bytes,output_bytes,mix_fnv64,left_fnv64,right_fnv64"
    );
    for (sr, seconds) in [(44100, 2), (48000, 2), (96000, 2), (48000, 10)] {
        let left = signal(sr, sr as usize * seconds, 4);
        let right = signal(sr, sr as usize * seconds, 2);
        let mut mono_codes = None;
        for joint in [false, true] {
            let started = std::time::Instant::now();
            let input = if joint {
                Input::Joint {
                    left: &left,
                    right: &right,
                }
            } else {
                Input::Mono(&left)
            };
            let out = analyze_input_codes(
                input,
                sr,
                &frequencies,
                &cfg,
                Limits::default(),
                &|| true,
                &|_| {},
                &code,
            )
            .unwrap();
            let elapsed = started.elapsed().as_secs_f64();
            println!(
                "cqt_probe,{},{sr},{seconds},{bars},{},{},{elapsed:.6},{},{},{},{},{},{},{},{:016x},{:016x},{:016x}",
                if joint { "joint" } else { "mono" },
                cfg.bins_per_octave,
                cfg.frame_rate,
                out.frames,
                out.work.kernel_ffts,
                out.work.input_ffts,
                out.work.inverse_ffts,
                out.work.batches,
                out.work.peak_workspace_bytes,
                out.work.output_bytes,
                digest(&out.codes[0]),
                digest(&out.codes[1]),
                digest(&out.codes[2])
            );
            assert_eq!(out.frames, seconds * 180);
            assert!(out.work.peak_workspace_bytes <= Limits::default().workspace_bytes);
            if joint {
                assert_eq!(out.codes[1], mono_codes.take().unwrap());
                assert_eq!(out.work.column_convolutions, 2 * bars as u64);
            } else {
                assert_eq!(out.work.column_convolutions, bars as u64);
                mono_codes = Some(out.codes[0].clone());
            }
        }
    }
}

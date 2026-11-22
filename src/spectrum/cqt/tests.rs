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

#[test]
fn sync_code_map_matches_dyn_code_map_bit_for_bit() {
    let frequencies = [
        180.0, 210.0, 250.0, 310.0, 440.0, 620.0, 880.0, 1200.0, 1800.0, 3200.0, 6000.0, 11000.0,
    ];
    let left = signal(48_000, 96_000, 4);
    let right = signal(48_000, 96_000, 2);
    let cfg = config(Window::Blackman);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    for input in [Input::Mono(&left), Input::Joint { left: &left, right: &right }] {
        let (true_de4fcb3_out, mod_ref_out, ref_out, sync_out) = pool.install(|| {
            let t = true_de4fcb3_analyze_input_codes(
                input,
                48_000,
                &frequencies,
                &cfg,
                Limits::default(),
                &|| true,
                &|_| {},
                &code,
            )
            .unwrap();
            let o = modified_serial_prep_ref_analyze_input_codes(
                input,
                48_000,
                &frequencies,
                &cfg,
                Limits::default(),
                &|| true,
                &|_| {},
                &code,
            )
            .unwrap();
            let a = analyze_input_codes(
                input,
                48_000,
                &frequencies,
                &cfg,
                Limits::default(),
                &|| true,
                &|_| {},
                &code,
            )
            .unwrap();
            let b = analyze_input_codes_sync(
                input,
                48_000,
                &frequencies,
                &cfg,
                Limits::default(),
                &|| true,
                &|_| {},
                &code,
            )
            .unwrap();
            (t, o, a, b)
        });
        assert_eq!(sync_out.codes, true_de4fcb3_out.codes);
        assert_eq!(sync_out.codes, mod_ref_out.codes);
        assert_eq!(sync_out.codes, ref_out.codes);
        assert_eq!(sync_out.frames, true_de4fcb3_out.frames);
        assert_eq!(sync_out.bars, true_de4fcb3_out.bars);
        assert_eq!(sync_out.work, ref_out.work);
        assert_work_counts_eq(&mod_ref_out.work, &true_de4fcb3_out.work);
        assert_work_counts_eq(&sync_out.work, &true_de4fcb3_out.work);
    }
}

fn assert_work_counts_eq(a: &WorkStats, b: &WorkStats) {
    assert_eq!(a.column_convolutions, b.column_convolutions);
    assert_eq!(a.kernel_ffts, b.kernel_ffts);
    assert!(a.input_ffts <= b.input_ffts, "new input_ffts {} > old {}", a.input_ffts, b.input_ffts);
    assert_eq!(a.inverse_ffts, b.inverse_ffts);
    assert_eq!(a.inverses_pruned, b.inverses_pruned);
    assert_eq!(a.inverses_full, b.inverses_full);
    assert_eq!(a.pruned_output_values, b.pruned_output_values);
    assert_eq!(a.pruning_workspace_fallbacks, b.pruning_workspace_fallbacks);
    assert_eq!(a.parallel_groups, b.parallel_groups);
    assert_eq!(a.max_inverse_workers, b.max_inverse_workers);
    assert_eq!(a.parallel_allocation_fallbacks, b.parallel_allocation_fallbacks);
    assert!(a.batches <= b.batches, "new batches {} > old {}", a.batches, b.batches);
    assert_eq!(a.output_bytes, b.output_bytes);
    assert!(a.peak_workspace_bytes <= Limits::default().workspace_bytes);
    assert!(b.peak_workspace_bytes <= Limits::default().workspace_bytes);
}

// ---------------------------------------------------------------------------
// 1. True unmodified `de4fcb3` implementation (`true_de4fcb3_parallel` +
//    `true_de4fcb3_analyze_input_codes`, including parallel `prepare_kernels`,
//    original `Plan`, `Slot`, `Workspace`, `old_de4fcb3_kernel`, and
//    `old_de4fcb3_fill_input`).
// 2. Modified Round-1 reference route (`modified_serial_prep_ref_analyze_input_codes`,
//    retained for historical traceability only; uses serial kernel prep).
// ---------------------------------------------------------------------------
fn old_de4fcb3_kernel(
    bin: Bin,
    n: usize,
    sr: u32,
    window: Window,
    should_continue: &dyn Fn() -> bool,
) -> Result<Vec<Complex<f64>>, Error> {
    let mut spectrum = zeroed::<Complex<f64>>(n)?;
    let mut sum = 0.0;
    for i in 0..=2 * bin.half {
        if i % 4096 == 0 {
            poll(should_continue)?;
        }
        let j = i as f64 - bin.half as f64;
        let w = window.at(PI * j / bin.half as f64);
        sum += w;
        let at = (n + bin.half - i) % n;
        spectrum[at].re = w;
    }
    let gain = 2.0 / sum;
    for i in 0..=2 * bin.half {
        if i % 4096 == 0 {
            poll(should_continue)?;
        }
        let j = i as f64 - bin.half as f64;
        let at = (n + bin.half - i) % n;
        let amplitude = gain * spectrum[at].re;
        let angle = -TAU * bin.frequency * j / f64::from(sr);
        spectrum[at] = Complex::new(amplitude * angle.cos(), amplitude * angle.sin());
    }
    Ok(spectrum)
}

fn old_de4fcb3_fill_input(
    target: &mut [Complex<f64>],
    source: &[f32],
    start: usize,
    half: usize,
    should_continue: &dyn Fn() -> bool,
) -> Result<(), Error> {
    for (i, value) in target.iter_mut().enumerate() {
        if i % 4096 == 0 {
            poll(should_continue)?;
        }
        let source_at = if i >= half {
            start.checked_add(i - half)
        } else {
            start.checked_sub(half - i)
        };
        let sample = source_at
            .and_then(|at| source.get(at))
            .copied()
            .unwrap_or(0.0);
        if !sample.is_finite() {
            return Err(Error::Invalid("CQT input contains a non-finite sample"));
        }
        *value = Complex::new(f64::from(sample), 0.0);
    }
    Ok(())
}

mod true_de4fcb3_parallel {
    use super::{
        Bin, Error, Limits, Requested, Window, checked_product, frame_center, old_de4fcb3_kernel,
        phases, poll, zeroed,
    };
    use rayon::prelude::*;
    use rustfft::{Fft, num_complex::Complex};
    use std::mem::size_of;

    type C64 = Complex<f64>;
    const MIN_PARALLEL_N: usize = 512;

    #[derive(Clone, Copy, Debug)]
    pub(super) struct Plan {
        pub slots: usize,
        pub batch_size: usize,
        pub workspace_bytes: usize,
        n: usize,
        channels: usize,
        scratch_len: usize,
        coefficient_cap: usize,
        requested_cap: Option<usize>,
        frames: usize,
        half: usize,
        sample_rate: u32,
        frame_rate: f32,
    }

    impl Plan {
        #[allow(clippy::too_many_arguments)]
        pub fn for_group(
            n: usize,
            channels: usize,
            bins: usize,
            frames: usize,
            half: usize,
            sample_rate: u32,
            frame_rate: f32,
            scratch_len: usize,
            inverse: &dyn Fft<f64>,
            limits: Limits,
            allow_pruning: bool,
        ) -> Option<Self> {
            rayon::current_thread_index()?;
            Self::with_workers(
                n,
                channels,
                bins,
                frames,
                half,
                sample_rate,
                frame_rate,
                scratch_len,
                inverse,
                limits,
                allow_pruning,
                rayon::current_num_threads(),
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn with_workers(
            n: usize,
            channels: usize,
            bins: usize,
            frames: usize,
            half: usize,
            sample_rate: u32,
            frame_rate: f32,
            scratch_len: usize,
            inverse: &dyn Fft<f64>,
            limits: Limits,
            allow_pruning: bool,
            workers: usize,
        ) -> Option<Self> {
            if n < MIN_PARALLEL_N
                || inverse.len() != n
                || !(1..=2).contains(&channels)
                || bins < 2
                || frames == 0
                || workers < 2
                || sample_rate == 0
                || !frame_rate.is_finite()
                || frame_rate <= 0.0
                || scratch_len < inverse.get_inplace_scratch_len()
            {
                return None;
            }
            let payload = n.checked_sub(half.checked_mul(2)?)?;
            if payload == 0 {
                return None;
            }
            let cap = ((payload as f64 * f64::from(frame_rate) / f64::from(sample_rate)).ceil()
                as usize)
                .saturating_add(1)
                .min(frames);
            let dense = n.checked_mul(size_of::<C64>())?;
            let sources = channels.checked_mul(dense)?;
            let product_and_scratch = channels
                .checked_mul(dense)?
                .checked_add(scratch_len.checked_mul(size_of::<C64>())?)?;
            let slab_per_bin = channels.checked_mul(cap)?.checked_mul(size_of::<C64>())?;
            let per_bin = dense.checked_add(slab_per_bin)?;
            let sparse_cap = cap.min(n / 4);
            for requested_cap in [
                (allow_pruning && inverse.prunes()).then_some(sparse_cap),
                None,
            ] {
                let requested = match requested_cap {
                    Some(c) => Requested::bytes(inverse, c, channels).ok()?,
                    None => 0,
                };
                let slot_bytes = product_and_scratch.checked_add(requested)?;
                for slots in (2..=workers.min(bins)).rev() {
                    let fixed = sources.checked_add(slots.checked_mul(slot_bytes)?)?;
                    let Some(available) = limits.workspace_bytes.checked_sub(fixed) else {
                        continue;
                    };
                    let batch_size = (available / per_bin).min(bins);
                    if batch_size < slots {
                        continue;
                    }
                    return Some(Self {
                        slots,
                        batch_size,
                        workspace_bytes: fixed.checked_add(batch_size.checked_mul(per_bin)?)?,
                        n,
                        channels,
                        scratch_len,
                        coefficient_cap: cap,
                        requested_cap,
                        frames,
                        half,
                        sample_rate,
                        frame_rate,
                    });
                }
            }
            None
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub(super) struct Block {
        pub first: usize,
        pub end: usize,
        pub sample_rate: u32,
        pub frame_rate: f32,
        pub half: usize,
        pub start: usize,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(super) struct Stats {
        pub inverse_ffts: u64,
        pub inverses_pruned: u64,
        pub inverses_full: u64,
        pub pruned_output_values: u64,
        pub worker_wall_ns: u128,
        pub worker_tasks: u64,
    }

    impl Stats {
        fn add(self, rhs: Self) -> Self {
            Self {
                inverse_ffts: self.inverse_ffts + rhs.inverse_ffts,
                inverses_pruned: self.inverses_pruned + rhs.inverses_pruned,
                inverses_full: self.inverses_full + rhs.inverses_full,
                pruned_output_values: self.pruned_output_values + rhs.pruned_output_values,
                worker_wall_ns: self.worker_wall_ns + rhs.worker_wall_ns,
                worker_tasks: self.worker_tasks + rhs.worker_tasks,
            }
        }
    }

    struct Slot {
        product_left: Vec<C64>,
        product_right: Vec<C64>,
        scratch: Vec<C64>,
        requested: Option<Requested>,
        stats: Stats,
    }

    impl Slot {
        fn new(plan: Plan, inverse: &dyn Fft<f64>) -> Result<Self, Error> {
            Ok(Self {
                product_left: zeroed(plan.n)?,
                product_right: zeroed(if plan.channels == 2 { plan.n } else { 0 })?,
                scratch: zeroed(plan.scratch_len)?,
                requested: plan
                    .requested_cap
                    .map(|c| Requested::new(inverse, c, plan.channels))
                    .transpose()?,
                stats: Stats::default(),
            })
        }

        fn prepare_kernels(
            &mut self,
            bins: &[Bin],
            destination: &mut [Vec<C64>],
            forward: &dyn Fft<f64>,
            plan: Plan,
            window: Window,
            should_continue: &(dyn Fn() -> bool + Sync),
        ) -> Result<u64, Error> {
            let mut entered = 0;
            for (&bin, out) in bins.iter().zip(destination.iter_mut()) {
                poll(should_continue)?;
                let mut values =
                    old_de4fcb3_kernel(bin, plan.n, plan.sample_rate, window, should_continue)?;
                poll(should_continue)?;
                entered += 1;
                forward.process_with_scratch(&mut values, &mut self.scratch);
                *out = values;
                poll(should_continue)?;
            }
            Ok(entered)
        }

        #[allow(clippy::too_many_arguments)]
        fn run(
            &mut self,
            kernels: &[Vec<C64>],
            destination: &mut [C64],
            source_left: &[C64],
            source_right: &[C64],
            inverse: &dyn Fft<f64>,
            plan: Plan,
            block: Block,
            should_continue: &(dyn Fn() -> bool + Sync),
            phase_timing: bool,
        ) -> Result<(), Error> {
            let phase = phases::clock(phase_timing);
            let count = block.end - block.first;
            let prune = self.requested.as_mut().is_some_and(|r| {
                r.prepare(
                    block.first,
                    block.end,
                    block.sample_rate,
                    block.frame_rate,
                    block.half,
                    block.start,
                )
            });
            let bin_stride = plan.channels * plan.coefficient_cap;
            for (kernel, out) in kernels.iter().zip(destination.chunks_exact_mut(bin_stride)) {
                poll(should_continue)?;
                if prune {
                    let r = self.requested.as_mut().expect("prepared above");
                    self.stats.inverse_ffts += 1;
                    self.stats.inverses_pruned += 1;
                    self.stats.pruned_output_values += count as u64;
                    r.outputs(
                        inverse,
                        source_left,
                        kernel,
                        &mut self.product_left,
                        &mut self.scratch,
                        count,
                        false,
                    );
                    if plan.channels == 2 {
                        poll(should_continue)?;
                        self.stats.inverse_ffts += 1;
                        self.stats.inverses_pruned += 1;
                        self.stats.pruned_output_values += count as u64;
                        r.outputs(
                            inverse,
                            source_right,
                            kernel,
                            &mut self.product_right,
                            &mut self.scratch,
                            count,
                            true,
                        );
                    }
                    out[..count].copy_from_slice(&r.left[..count]);
                    if plan.channels == 2 {
                        out[plan.coefficient_cap..plan.coefficient_cap + count]
                            .copy_from_slice(&r.right[..count]);
                    }
                } else {
                    for i in 0..plan.n {
                        if i % 4096 == 0 {
                            poll(should_continue)?;
                        }
                        self.product_left[i] = source_left[i] * kernel[i];
                        if plan.channels == 2 {
                            self.product_right[i] = source_right[i] * kernel[i];
                        }
                    }
                    self.stats.inverse_ffts += 1;
                    self.stats.inverses_full += 1;
                    inverse.process_with_scratch(&mut self.product_left, &mut self.scratch);
                    if plan.channels == 2 {
                        poll(should_continue)?;
                        self.stats.inverse_ffts += 1;
                        self.stats.inverses_full += 1;
                        inverse.process_with_scratch(&mut self.product_right, &mut self.scratch);
                    }
                    for (offset, frame) in (block.first..block.end).enumerate() {
                        let at = block.half
                            + frame_center(frame, block.sample_rate, block.frame_rate)
                            - block.start;
                        out[offset] = self.product_left[at];
                        if plan.channels == 2 {
                            out[plan.coefficient_cap + offset] = self.product_right[at];
                        }
                    }
                }
            }
            if phase_timing {
                self.stats.worker_wall_ns += phases::elapsed(phase);
                self.stats.worker_tasks += 1;
            }
            Ok(())
        }
    }

    pub(super) struct Workspace {
        plan: Plan,
        source_left: Vec<C64>,
        source_right: Vec<C64>,
        slots: Vec<Slot>,
        coefficients: Vec<C64>,
        active_bins: usize,
        active_count: usize,
        phase_timing: bool,
    }

    impl Workspace {
        pub fn new(plan: Plan, inverse: &dyn Fft<f64>) -> Result<Self, Error> {
            let mut slots = Vec::new();
            slots
                .try_reserve_exact(plan.slots)
                .map_err(|_| Error::Allocation)?;
            for _ in 0..plan.slots {
                slots.push(Slot::new(plan, inverse)?);
            }
            Ok(Self {
                plan,
                source_left: zeroed(plan.n)?,
                source_right: zeroed(if plan.channels == 2 { plan.n } else { 0 })?,
                slots,
                coefficients: zeroed(checked_product(
                    checked_product(plan.batch_size, plan.channels)?,
                    plan.coefficient_cap,
                )?)?,
                active_bins: 0,
                active_count: 0,
                phase_timing: false,
            })
        }

        pub fn set_phase_timing(&mut self, enabled: bool) {
            self.phase_timing = enabled;
        }

        pub fn source_buffers(&mut self) -> (&mut [C64], &mut [C64], &mut [C64]) {
            (
                &mut self.source_left,
                &mut self.source_right,
                &mut self.slots[0].scratch,
            )
        }

        pub fn prepare_kernels(
            &mut self,
            bins: &[Bin],
            forward: &dyn Fft<f64>,
            window: Window,
            should_continue: &(dyn Fn() -> bool + Sync),
        ) -> Result<(Vec<Vec<C64>>, u64), Error> {
            self.active_bins = 0;
            self.active_count = 0;
            poll(should_continue)?;
            if bins.is_empty()
                || bins.len() > self.plan.batch_size
                || forward.len() != self.plan.n
                || forward.fft_direction() != rustfft::FftDirection::Forward
                || self.slots.len() != self.plan.slots
                || self
                    .slots
                    .iter()
                    .any(|slot| slot.scratch.len() < forward.get_inplace_scratch_len())
                || bins.iter().any(|bin| {
                    bin.half > self.plan.half
                        || !bin.frequency.is_finite()
                        || bin.frequency <= 0.0
                        || bin.frequency >= f64::from(self.plan.sample_rate) * 0.5
                })
            {
                return Err(Error::Invalid(
                    "CQT kernel preparation does not match its admitted workspace",
                ));
            }
            let mut kernels = Vec::new();
            kernels
                .try_reserve_exact(bins.len())
                .map_err(|_| Error::Allocation)?;
            kernels.resize_with(bins.len(), Vec::new);
            let chunk = bins.len().div_ceil(self.plan.slots);
            let plan = self.plan;
            let entered = if rayon::current_thread_index().is_some() && self.plan.slots > 1 {
                self.slots
                    .par_iter_mut()
                    .zip(bins.par_chunks(chunk))
                    .zip(kernels.par_chunks_mut(chunk))
                    .map(|((slot, bins), destination)| {
                        slot.prepare_kernels(
                            bins,
                            destination,
                            forward,
                            plan,
                            window,
                            should_continue,
                        )
                    })
                    .try_reduce(|| 0, |left, right| Ok(left + right))?
            } else {
                let mut entered = 0;
                for ((slot, bins), destination) in self
                    .slots
                    .iter_mut()
                    .zip(bins.chunks(chunk))
                    .zip(kernels.chunks_mut(chunk))
                {
                    entered += slot.prepare_kernels(
                        bins,
                        destination,
                        forward,
                        plan,
                        window,
                        should_continue,
                    )?;
                }
                entered
            };
            poll(should_continue)?;
            Ok((kernels, entered))
        }

        pub fn process_block(
            &mut self,
            kernels: &[Vec<C64>],
            inverse: &dyn Fft<f64>,
            block: Block,
            should_continue: &(dyn Fn() -> bool + Sync),
        ) -> Result<Stats, Error> {
            self.active_bins = 0;
            self.active_count = 0;
            poll(should_continue)?;
            let count = block
                .end
                .checked_sub(block.first)
                .ok_or(Error::Invalid("CQT block runs backwards"))?;
            if count == 0
                || block.end > self.plan.frames
                || count > self.plan.coefficient_cap
                || kernels.is_empty()
                || kernels.len() > self.plan.batch_size
                || kernels.iter().any(|h| h.len() != self.plan.n)
                || inverse.len() != self.plan.n
                || block.half != self.plan.half
                || block.sample_rate != self.plan.sample_rate
                || block.frame_rate.to_bits() != self.plan.frame_rate.to_bits()
                || block.start != frame_center(block.first, block.sample_rate, block.frame_rate)
            {
                return Err(Error::Invalid("CQT parallel block does not match its plan"));
            }
            let last = block
                .half
                .checked_add(
                    frame_center(block.end - 1, block.sample_rate, block.frame_rate)
                        .checked_sub(block.start)
                        .ok_or(Error::Invalid("CQT block center precedes its input"))?,
                )
                .ok_or(Error::Resource("CQT block center overflows"))?;
            if last >= self.plan.n {
                return Err(Error::Invalid(
                    "CQT requested sample is outside its transform",
                ));
            }
            for slot in &mut self.slots {
                slot.stats = Stats::default();
            }
            let bin_stride = self.plan.channels * self.plan.coefficient_cap;
            let chunk_bins = kernels.len().div_ceil(self.plan.slots);
            let chunk_stride = chunk_bins * bin_stride;
            let slab = &mut self.coefficients[..kernels.len() * bin_stride];
            let plan = self.plan;
            let left = &self.source_left;
            let right = &self.source_right;
            let phase_timing = self.phase_timing;
            if rayon::current_thread_index().is_some() && self.plan.slots > 1 {
                self.slots
                    .par_iter_mut()
                    .zip(slab.par_chunks_mut(chunk_stride))
                    .enumerate()
                    .try_for_each(|(index, (slot, out))| {
                        let begin = index * chunk_bins;
                        let end = (begin + chunk_bins).min(kernels.len());
                        slot.run(
                            &kernels[begin..end],
                            out,
                            left,
                            right,
                            inverse,
                            plan,
                            block,
                            should_continue,
                            phase_timing,
                        )
                    })?;
            } else {
                for (index, (slot, out)) in self
                    .slots
                    .iter_mut()
                    .zip(slab.chunks_mut(chunk_stride))
                    .enumerate()
                {
                    let begin = index * chunk_bins;
                    let end = (begin + chunk_bins).min(kernels.len());
                    slot.run(
                        &kernels[begin..end],
                        out,
                        left,
                        right,
                        inverse,
                        plan,
                        block,
                        should_continue,
                        phase_timing,
                    )?;
                }
            }
            poll(should_continue)?;
            self.active_bins = kernels.len();
            self.active_count = count;
            Ok(self
                .slots
                .iter()
                .fold(Stats::default(), |s, slot| s.add(slot.stats)))
        }

        pub fn coefficients(&self, bin: usize, channel: usize, count: usize) -> &[C64] {
            assert!(
                bin < self.active_bins
                    && channel < self.plan.channels
                    && count <= self.active_count
            );
            let at = (bin * self.plan.channels + channel) * self.plan.coefficient_cap;
            &self.coefficients[at..at + count]
        }
    }
}

/// Verbatim unmodified `de4fcb3` `analyze_input_codes` (using `true_de4fcb3_parallel`
/// with parallel `workspace.prepare_kernels`).
#[allow(clippy::too_many_arguments)]
fn true_de4fcb3_analyze_input_codes(
    input: Input<'_>,
    sample_rate: u32,
    frequencies: &[f64],
    config: &Config,
    limits: Limits,
    should_continue: &(dyn Fn() -> bool + Sync),
    on_progress: &dyn Fn(f32),
    map: &dyn Fn(f32) -> u16,
) -> Result<Output, Error> {
    let phase_timing = false;
    let mut profile = phases::Profile::new(phase_timing);
    let phase = profile.clock();
    poll(should_continue)?;
    let groups = layout(frequencies, sample_rate, config, limits)?;
    let (left, right) = match input {
        Input::Mono(signal) => (signal, None),
        Input::Joint { left, right } => {
            if left.len() != right.len() {
                return Err(Error::Invalid(
                    "CQT stereo channels must have the same length",
                ));
            }
            (left, Some(right))
        }
    };
    let frames = frame_count(left.len(), sample_rate, config.frame_rate)?;
    if frames == 0 {
        return Err(Error::Invalid("CQT input is empty"));
    }
    let bars = frequencies.len();
    let cells = checked_product(frames, bars)?;
    let channels = if right.is_some() { 2 } else { 1 };
    let planes = if right.is_some() { 3 } else { 1 };
    let output_bytes = checked_product(checked_product(cells, planes)?, size_of::<u16>())?;
    if output_bytes > limits.output_bytes {
        return Err(Error::Resource("returned code planes exceed output_bytes"));
    }
    for &n in groups.keys() {
        let dense_bytes = checked_product(n, size_of::<Complex<f64>>())?;
        let minimum = checked_product(dense_bytes, channels * 2 + 1)?;
        if minimum > limits.workspace_bytes {
            return Err(Error::Resource(
                "one full kernel and source/product buffers exceed workspace_bytes",
            ));
        }
    }
    let mut codes = [zeroed(cells)?, Vec::new(), Vec::new()];
    if right.is_some() {
        codes[1] = zeroed(cells)?;
        codes[2] = zeroed(cells)?;
    }
    let mut work = WorkStats {
        output_bytes,
        ..WorkStats::default()
    };
    profile.record(phases::Region::Setup, phase);
    let mut completed = 0usize;
    on_progress(0.0);
    for (n, bins) in groups {
        let group_phase = profile.begin_group(n, bins.len(), work);
        poll(should_continue)?;
        let dense_bytes = checked_product(n, size_of::<Complex<f64>>())?;
        let base_vectors = channels * 2;
        let minimum = checked_product(dense_bytes, base_vectors + 1)?;
        if minimum > limits.workspace_bytes {
            return Err(Error::Resource(
                "one full kernel and source/product buffers exceed workspace_bytes",
            ));
        }
        let phase = profile.clock();
        let mut planner = FftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(n);
        let inverse = planner.plan_fft_inverse(n);
        let scratch_len = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        let base_bytes = checked_product(dense_bytes, base_vectors)?
            .checked_add(checked_product(scratch_len, size_of::<Complex<f64>>())?)
            .ok_or(Error::Resource("workspace size overflows"))?;
        let half = bins
            .iter()
            .map(|bin| bin.half)
            .max()
            .expect("a nonempty group");
        let payload = n - 2 * half;
        profile.record(phases::Region::Planning, phase);
        let allow_pruning = true;
        let phase = profile.clock();
        let parallel_plan = if allow_pruning {
            true_de4fcb3_parallel::Plan::for_group(
                n,
                channels,
                bins.len(),
                frames,
                half,
                sample_rate,
                config.frame_rate,
                scratch_len,
                &*inverse,
                limits,
                true,
            )
        } else {
            None
        };
        profile.record(phases::Region::Planning, phase);
        if let Some(plan) = parallel_plan {
            let phase = profile.clock();
            let workspace = true_de4fcb3_parallel::Workspace::new(plan, &*inverse);
            profile.record(phases::Region::Allocation, phase);
            match workspace {
                Ok(mut workspace) => {
                    workspace.set_phase_timing(phase_timing);
                    work.parallel_groups += 1;
                    work.max_inverse_workers = work.max_inverse_workers.max(plan.slots);
                    work.peak_workspace_bytes =
                        work.peak_workspace_bytes.max(plan.workspace_bytes);
                    for batch in bins.chunks(plan.batch_size) {
                        poll(should_continue)?;
                        work.batches += 1;
                        let phase = profile.clock();
                        let (kernels, entered) = workspace.prepare_kernels(
                            batch,
                            &*forward,
                            config.window,
                            should_continue,
                        )?;
                        work.kernel_ffts += entered;
                        profile.record(phases::Region::Kernel, phase);
                        let mut first = 0;
                        while first < frames {
                            poll(should_continue)?;
                            let start = frame_center(first, sample_rate, config.frame_rate);
                            let mut end = first + 1;
                            while end < frames
                                && frame_center(end, sample_rate, config.frame_rate) - start
                                    < payload
                            {
                                end += 1;
                            }
                            let phase = profile.clock();
                            {
                                let (source_left, source_right, scratch) =
                                    workspace.source_buffers();
                                old_de4fcb3_fill_input(
                                    source_left,
                                    left,
                                    start,
                                    half,
                                    should_continue,
                                )?;
                                work.input_ffts += 1;
                                forward.process_with_scratch(source_left, scratch);
                                if let Some(right) = right {
                                    old_de4fcb3_fill_input(
                                        source_right,
                                        right,
                                        start,
                                        half,
                                        should_continue,
                                    )?;
                                    work.input_ffts += 1;
                                    forward.process_with_scratch(source_right, scratch);
                                }
                            }
                            profile.record(phases::Region::Source, phase);
                            let phase = profile.clock();
                            let stats = workspace.process_block(
                                &kernels,
                                &*inverse,
                                true_de4fcb3_parallel::Block {
                                    first,
                                    end,
                                    sample_rate,
                                    frame_rate: config.frame_rate,
                                    half,
                                    start,
                                },
                                should_continue,
                            )?;
                            profile.record(phases::Region::ParallelInverse, phase);
                            profile.workers(stats.worker_wall_ns, stats.worker_tasks);
                            work.inverse_ffts += stats.inverse_ffts;
                            work.inverses_pruned += stats.inverses_pruned;
                            work.inverses_full += stats.inverses_full;
                            work.pruned_output_values += stats.pruned_output_values;
                            let phase = profile.clock();
                            let norm = 1.0 / n as f64;
                            for (local, bin) in batch.iter().enumerate() {
                                poll(should_continue)?;
                                if first == 0 {
                                    work.column_convolutions += channels as u64;
                                }
                                let ls = workspace.coefficients(local, 0, end - first);
                                let rs =
                                    right.map(|_| workspace.coefficients(local, 1, end - first));
                                for frame in first..end {
                                    let l = ls[frame - first] * norm;
                                    let r = rs.map(|values| values[frame - first] * norm);
                                    let cell = frame * bars + bin.index;
                                    if let Some(r) = r {
                                        codes[0][cell] = map(((l + r) * 0.5).norm() as f32);
                                        codes[1][cell] = map(l.norm() as f32);
                                        codes[2][cell] = map(r.norm() as f32);
                                    } else {
                                        codes[0][cell] = map(l.norm() as f32);
                                    }
                                }
                                completed += end - first;
                                on_progress(completed as f32 / cells as f32);
                            }
                            profile.record(phases::Region::ParallelFold, phase);
                            first = end;
                        }
                    }
                    profile.finish_group(
                        group_phase,
                        plan.slots,
                        plan.batch_size,
                        plan.workspace_bytes,
                        work,
                    );
                    continue;
                }
                Err(Error::Allocation) => {
                    work.parallel_allocation_fallbacks += 1;
                }
                Err(error) => return Err(error),
            }
        }
        let mut requested = None;
        let mut requested_bytes = 0;
        if allow_pruning && inverse.prunes() {
            let cap = requested_cap(payload, n, frames, sample_rate, config.frame_rate);
            let bytes = Requested::bytes(&*inverse, cap, channels)?;
            let admitted = base_bytes
                .checked_add(dense_bytes)
                .and_then(|minimum| minimum.checked_add(bytes))
                .is_some_and(|minimum| minimum <= limits.workspace_bytes);
            if admitted {
                if let Ok(buffers) = Requested::new(&*inverse, cap, channels) {
                    requested = Some(buffers);
                    requested_bytes = bytes;
                } else {
                    work.pruning_workspace_fallbacks += 1;
                }
            } else {
                work.pruning_workspace_fallbacks += 1;
            }
        }
        let base_bytes = base_bytes
            .checked_add(requested_bytes)
            .ok_or(Error::Resource("requested-output workspace overflows"))?;
        let available = limits
            .workspace_bytes
            .checked_sub(base_bytes)
            .ok_or(Error::Resource("FFT scratch exceeds workspace_bytes"))?;
        let batch_size = (available / dense_bytes).min(bins.len());
        if batch_size == 0 {
            return Err(Error::Resource(
                "FFT scratch leaves no room for a full kernel",
            ));
        }
        let mut source_left = zeroed::<Complex<f64>>(n)?;
        let mut source_right = zeroed::<Complex<f64>>(if right.is_some() { n } else { 0 })?;
        let mut product_left = zeroed::<Complex<f64>>(n)?;
        let mut product_right = zeroed::<Complex<f64>>(if right.is_some() { n } else { 0 })?;
        let mut scratch = zeroed::<Complex<f64>>(scratch_len)?;
        for batch in bins.chunks(batch_size) {
            poll(should_continue)?;
            work.batches += 1;
            work.peak_workspace_bytes = work
                .peak_workspace_bytes
                .max(base_bytes + batch.len() * dense_bytes);
            let mut kernels = Vec::new();
            kernels
                .try_reserve_exact(batch.len())
                .map_err(|_| Error::Allocation)?;
            for &bin in batch {
                let mut spectrum =
                    old_de4fcb3_kernel(bin, n, sample_rate, config.window, should_continue)?;
                poll(should_continue)?;
                work.kernel_ffts += 1;
                forward.process_with_scratch(&mut spectrum, &mut scratch);
                kernels.push(spectrum);
            }
            let mut first = 0;
            while first < frames {
                poll(should_continue)?;
                let start = frame_center(first, sample_rate, config.frame_rate);
                let mut end = first + 1;
                while end < frames
                    && frame_center(end, sample_rate, config.frame_rate) - start < payload
                {
                    end += 1;
                }
                let prune = requested.as_mut().is_some_and(|buffers| {
                    buffers.prepare(first, end, sample_rate, config.frame_rate, half, start)
                });
                old_de4fcb3_fill_input(&mut source_left, left, start, half, should_continue)?;
                work.input_ffts += 1;
                forward.process_with_scratch(&mut source_left, &mut scratch);
                if let Some(right) = right {
                    old_de4fcb3_fill_input(
                        &mut source_right,
                        right,
                        start,
                        half,
                        should_continue,
                    )?;
                    work.input_ffts += 1;
                    forward.process_with_scratch(&mut source_right, &mut scratch);
                }
                for (&bin, spectrum) in batch.iter().zip(&kernels) {
                    poll(should_continue)?;
                    if first == 0 {
                        work.column_convolutions += channels as u64;
                    }
                    if prune {
                        let buffers = requested.as_mut().expect("a prepared request");
                        work.inverse_ffts += 1;
                        work.inverses_pruned += 1;
                        work.pruned_output_values += (end - first) as u64;
                        buffers.outputs(
                            &*inverse,
                            &source_left,
                            spectrum,
                            &mut product_left,
                            &mut scratch,
                            end - first,
                            false,
                        );
                        if right.is_some() {
                            poll(should_continue)?;
                            work.inverse_ffts += 1;
                            work.inverses_pruned += 1;
                            work.pruned_output_values += (end - first) as u64;
                            buffers.outputs(
                                &*inverse,
                                &source_right,
                                spectrum,
                                &mut product_right,
                                &mut scratch,
                                end - first,
                                true,
                            );
                        }
                    } else {
                        for i in 0..n {
                            if i % 4096 == 0 {
                                poll(should_continue)?;
                            }
                            product_left[i] = source_left[i] * spectrum[i];
                            if right.is_some() {
                                product_right[i] = source_right[i] * spectrum[i];
                            }
                        }
                        work.inverse_ffts += 1;
                        work.inverses_full += 1;
                        inverse.process_with_scratch(&mut product_left, &mut scratch);
                        if right.is_some() {
                            poll(should_continue)?;
                            work.inverse_ffts += 1;
                            work.inverses_full += 1;
                            inverse.process_with_scratch(&mut product_right, &mut scratch);
                        }
                    }
                    let norm = 1.0 / n as f64;
                    for frame in first..end {
                        let at = half + frame_center(frame, sample_rate, config.frame_rate) - start;
                        let l = (if prune {
                            requested.as_ref().expect("a prepared request").left[frame - first]
                        } else {
                            product_left[at]
                        }) * norm;
                        let r = right.map(|_| {
                            (if prune {
                                requested.as_ref().expect("a prepared request").right[frame - first]
                            } else {
                                product_right[at]
                            }) * norm
                        });
                        let cell = frame * bars + bin.index;
                        if let Some(r) = r {
                            codes[0][cell] = map(((l + r) * 0.5).norm() as f32);
                            codes[1][cell] = map(l.norm() as f32);
                            codes[2][cell] = map(r.norm() as f32);
                        } else {
                            codes[0][cell] = map(l.norm() as f32);
                        }
                    }
                    completed += end - first;
                    on_progress(completed as f32 / cells as f32);
                }
                first = end;
            }
        }
    }
    poll(should_continue)?;
    work.phases = profile.finish();
    Ok(Output {
        codes,
        frames,
        bars,
        work,
    })
}

/// Round-1 modified reference route (uses serial kernel preparation and new
/// `parallel::Workspace`; retained for historical record per Codex Round 2 #2).
#[allow(clippy::too_many_arguments)]
fn modified_serial_prep_ref_analyze_input_codes(
    input: Input<'_>,
    sample_rate: u32,
    frequencies: &[f64],
    config: &Config,
    limits: Limits,
    should_continue: &(dyn Fn() -> bool + Sync),
    on_progress: &dyn Fn(f32),
    map: &dyn Fn(f32) -> u16,
) -> Result<Output, Error> {
    let phase_timing = false;
    let mut profile = phases::Profile::new(phase_timing);
    let phase = profile.clock();
    poll(should_continue)?;
    let groups = layout(frequencies, sample_rate, config, limits)?;
    let (left, right) = match input {
        Input::Mono(signal) => (signal, None),
        Input::Joint { left, right } => {
            if left.len() != right.len() {
                return Err(Error::Invalid(
                    "CQT stereo channels must have the same length",
                ));
            }
            (left, Some(right))
        }
    };
    let frames = frame_count(left.len(), sample_rate, config.frame_rate)?;
    if frames == 0 {
        return Err(Error::Invalid("CQT input is empty"));
    }
    let bars = frequencies.len();
    let cells = checked_product(frames, bars)?;
    let channels = if right.is_some() { 2 } else { 1 };
    let planes = if right.is_some() { 3 } else { 1 };
    let output_bytes = checked_product(checked_product(cells, planes)?, size_of::<u16>())?;
    if output_bytes > limits.output_bytes {
        return Err(Error::Resource("returned code planes exceed output_bytes"));
    }
    for &n in groups.keys() {
        let dense_bytes = checked_product(n, size_of::<Complex<f64>>())?;
        let minimum = checked_product(dense_bytes, channels * 2 + 1)?;
        if minimum > limits.workspace_bytes {
            return Err(Error::Resource(
                "one full kernel and source/product buffers exceed workspace_bytes",
            ));
        }
    }
    let mut codes = [zeroed(cells)?, Vec::new(), Vec::new()];
    if right.is_some() {
        codes[1] = zeroed(cells)?;
        codes[2] = zeroed(cells)?;
    }
    let mut work = WorkStats {
        output_bytes,
        ..WorkStats::default()
    };
    profile.record(phases::Region::Setup, phase);
    let mut completed = 0usize;
    on_progress(0.0);
    for (n, bins) in groups {
        let group_phase = profile.begin_group(n, bins.len(), work);
        poll(should_continue)?;
        let dense_bytes = checked_product(n, size_of::<Complex<f64>>())?;
        let base_vectors = channels * 2;
        let minimum = checked_product(dense_bytes, base_vectors + 1)?;
        if minimum > limits.workspace_bytes {
            return Err(Error::Resource(
                "one full kernel and source/product buffers exceed workspace_bytes",
            ));
        }
        let phase = profile.clock();
        let mut planner = FftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(n);
        let inverse = planner.plan_fft_inverse(n);
        let scratch_len = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        let base_bytes = checked_product(dense_bytes, base_vectors)?
            .checked_add(checked_product(scratch_len, size_of::<Complex<f64>>())?)
            .ok_or(Error::Resource("workspace size overflows"))?;
        let half = bins
            .iter()
            .map(|bin| bin.half)
            .max()
            .expect("a nonempty group");
        let payload = n - 2 * half;
        profile.record(phases::Region::Planning, phase);
        let allow_pruning = true;
        let phase = profile.clock();
        let parallel_plan = if allow_pruning {
            parallel::Plan::for_group(
                n,
                channels,
                bins.len(),
                frames,
                half,
                sample_rate,
                config.frame_rate,
                scratch_len,
                &*inverse,
                limits,
                true,
            )
        } else {
            None
        };
        profile.record(phases::Region::Planning, phase);
        if let Some(plan) = parallel_plan {
            let phase = profile.clock();
            let workspace = parallel::Workspace::new(plan, &*inverse);
            profile.record(phases::Region::Allocation, phase);
            match workspace {
                Ok(mut workspace) => {
                    workspace.set_phase_timing(phase_timing);
                    work.parallel_groups += 1;
                    work.max_inverse_workers = work.max_inverse_workers.max(plan.slots);
                    work.peak_workspace_bytes =
                        work.peak_workspace_bytes.max(plan.workspace_bytes);
                    for batch in bins.chunks(plan.batch_size) {
                        poll(should_continue)?;
                        work.batches += 1;
                        let phase = profile.clock();
                        let mut kernels = Vec::new();
                        kernels.try_reserve_exact(batch.len()).map_err(|_| Error::Allocation)?;
                        let mut k_scratch = zeroed::<Complex<f64>>(scratch_len)?;
                        for &bin in batch {
                            let mut sp = old_de4fcb3_kernel(bin, n, sample_rate, config.window, should_continue)?;
                            poll(should_continue)?;
                            work.kernel_ffts += 1;
                            forward.process_with_scratch(&mut sp, &mut k_scratch);
                            kernels.push(sp);
                        }
                        profile.record(phases::Region::Kernel, phase);
                        let mut first = 0;
                        while first < frames {
                            poll(should_continue)?;
                            let start = frame_center(first, sample_rate, config.frame_rate);
                            let mut end = first + 1;
                            while end < frames
                                && frame_center(end, sample_rate, config.frame_rate) - start
                                    < payload
                            {
                                end += 1;
                            }
                            let phase = profile.clock();
                            {
                                let (source_left, source_right, scratch) =
                                    workspace.source_buffers();
                                old_de4fcb3_fill_input(source_left, left, start, half, should_continue)?;
                                work.input_ffts += 1;
                                forward.process_with_scratch(source_left, scratch);
                                if let Some(right) = right {
                                    old_de4fcb3_fill_input(source_right, right, start, half, should_continue)?;
                                    work.input_ffts += 1;
                                    forward.process_with_scratch(source_right, scratch);
                                }
                            }
                            profile.record(phases::Region::Source, phase);
                            let phase = profile.clock();
                            let stats = workspace.process_block(
                                &kernels,
                                &*inverse,
                                parallel::Block {
                                    first,
                                    end,
                                    sample_rate,
                                    frame_rate: config.frame_rate,
                                    half,
                                    start,
                                },
                                should_continue,
                            )?;
                            profile.record(phases::Region::ParallelInverse, phase);
                            profile.workers(stats.worker_wall_ns, stats.worker_tasks);
                            work.inverse_ffts += stats.inverse_ffts;
                            work.inverses_pruned += stats.inverses_pruned;
                            work.inverses_full += stats.inverses_full;
                            work.pruned_output_values += stats.pruned_output_values;
                            let phase = profile.clock();
                            let norm = 1.0 / n as f64;
                            for (local, bin) in batch.iter().enumerate() {
                                poll(should_continue)?;
                                if first == 0 {
                                    work.column_convolutions += channels as u64;
                                }
                                let ls = workspace.coefficients(local, 0, end - first);
                                let rs =
                                    right.map(|_| workspace.coefficients(local, 1, end - first));
                                for frame in first..end {
                                    let l = ls[frame - first] * norm;
                                    let r = rs.map(|values| values[frame - first] * norm);
                                    let cell = frame * bars + bin.index;
                                    if let Some(r) = r {
                                        codes[0][cell] = map(((l + r) * 0.5).norm() as f32);
                                        codes[1][cell] = map(l.norm() as f32);
                                        codes[2][cell] = map(r.norm() as f32);
                                    } else {
                                        codes[0][cell] = map(l.norm() as f32);
                                    }
                                }
                                completed += end - first;
                                on_progress(completed as f32 / cells as f32);
                            }
                            profile.record(phases::Region::ParallelFold, phase);
                            first = end;
                        }
                    }
                    profile.finish_group(group_phase, plan.slots, plan.batch_size, plan.workspace_bytes, work);
                    continue;
                }
                Err(Error::Allocation) => {
                    work.parallel_allocation_fallbacks += 1;
                }
                Err(error) => return Err(error),
            }
        }
        let mut requested = None;
        let mut requested_bytes = 0;
        if allow_pruning && inverse.prunes() {
            let cap = requested_cap(payload, n, frames, sample_rate, config.frame_rate);
            let bytes = Requested::bytes(&*inverse, cap, channels)?;
            let admitted = base_bytes
                .checked_add(dense_bytes)
                .and_then(|minimum| minimum.checked_add(bytes))
                .is_some_and(|minimum| minimum <= limits.workspace_bytes);
            if admitted {
                if let Ok(buffers) = Requested::new(&*inverse, cap, channels) {
                    requested = Some(buffers);
                    requested_bytes = bytes;
                } else {
                    work.pruning_workspace_fallbacks += 1;
                }
            } else {
                work.pruning_workspace_fallbacks += 1;
            }
        }
        let base_bytes = base_bytes
            .checked_add(requested_bytes)
            .ok_or(Error::Resource("requested-output workspace overflows"))?;
        let available = limits
            .workspace_bytes
            .checked_sub(base_bytes)
            .ok_or(Error::Resource("FFT scratch exceeds workspace_bytes"))?;
        let batch_size = (available / dense_bytes).min(bins.len());
        if batch_size == 0 {
            return Err(Error::Resource(
                "FFT scratch leaves no room for a full kernel",
            ));
        }
        let mut source_left = zeroed::<Complex<f64>>(n)?;
        let mut source_right = zeroed::<Complex<f64>>(if right.is_some() { n } else { 0 })?;
        let mut product_left = zeroed::<Complex<f64>>(n)?;
        let mut product_right = zeroed::<Complex<f64>>(if right.is_some() { n } else { 0 })?;
        let mut scratch = zeroed::<Complex<f64>>(scratch_len)?;
        for batch in bins.chunks(batch_size) {
            poll(should_continue)?;
            work.batches += 1;
            work.peak_workspace_bytes = work
                .peak_workspace_bytes
                .max(base_bytes + batch.len() * dense_bytes);
            let mut kernels = Vec::new();
            kernels
                .try_reserve_exact(batch.len())
                .map_err(|_| Error::Allocation)?;
            for &bin in batch {
                let mut spectrum = old_de4fcb3_kernel(bin, n, sample_rate, config.window, should_continue)?;
                poll(should_continue)?;
                work.kernel_ffts += 1;
                forward.process_with_scratch(&mut spectrum, &mut scratch);
                kernels.push(spectrum);
            }
            let mut first = 0;
            while first < frames {
                poll(should_continue)?;
                let start = frame_center(first, sample_rate, config.frame_rate);
                let mut end = first + 1;
                while end < frames
                    && frame_center(end, sample_rate, config.frame_rate) - start < payload
                {
                    end += 1;
                }
                let prune = requested.as_mut().is_some_and(|buffers| {
                    buffers.prepare(first, end, sample_rate, config.frame_rate, half, start)
                });
                old_de4fcb3_fill_input(&mut source_left, left, start, half, should_continue)?;
                work.input_ffts += 1;
                forward.process_with_scratch(&mut source_left, &mut scratch);
                if let Some(right) = right {
                    old_de4fcb3_fill_input(&mut source_right, right, start, half, should_continue)?;
                    work.input_ffts += 1;
                    forward.process_with_scratch(&mut source_right, &mut scratch);
                }
                for (&bin, spectrum) in batch.iter().zip(&kernels) {
                    poll(should_continue)?;
                    if first == 0 {
                        work.column_convolutions += channels as u64;
                    }
                    if prune {
                        let buffers = requested.as_mut().expect("a prepared request");
                        work.inverse_ffts += 1;
                        work.inverses_pruned += 1;
                        work.pruned_output_values += (end - first) as u64;
                        buffers.outputs(
                            &*inverse,
                            &source_left,
                            spectrum,
                            &mut product_left,
                            &mut scratch,
                            end - first,
                            false,
                        );
                        if right.is_some() {
                            poll(should_continue)?;
                            work.inverse_ffts += 1;
                            work.inverses_pruned += 1;
                            work.pruned_output_values += (end - first) as u64;
                            buffers.outputs(
                                &*inverse,
                                &source_right,
                                spectrum,
                                &mut product_right,
                                &mut scratch,
                                end - first,
                                true,
                            );
                        }
                    } else {
                        for i in 0..n {
                            if i % 4096 == 0 {
                                poll(should_continue)?;
                            }
                            product_left[i] = source_left[i] * spectrum[i];
                            if right.is_some() {
                                product_right[i] = source_right[i] * spectrum[i];
                            }
                        }
                        work.inverse_ffts += 1;
                        work.inverses_full += 1;
                        inverse.process_with_scratch(&mut product_left, &mut scratch);
                        if right.is_some() {
                            poll(should_continue)?;
                            work.inverse_ffts += 1;
                            work.inverses_full += 1;
                            inverse.process_with_scratch(&mut product_right, &mut scratch);
                        }
                    }
                    let norm = 1.0 / n as f64;
                    for frame in first..end {
                        let at = half + frame_center(frame, sample_rate, config.frame_rate) - start;
                        let l = if prune {
                            requested.as_ref().expect("a prepared request").left[frame - first]
                        } else {
                            product_left[at]
                        } * norm;
                        let r = right.map(|_| if prune {
                            requested.as_ref().expect("a prepared request").right[frame - first]
                        } else { product_right[at] } * norm);
                        let cell = frame * bars + bin.index;
                        if let Some(r) = r {
                            codes[0][cell] = map(((l + r) * 0.5).norm() as f32);
                            codes[1][cell] = map(l.norm() as f32);
                            codes[2][cell] = map(r.norm() as f32);
                        } else {
                            codes[0][cell] = map(l.norm() as f32);
                        }
                    }
                    completed += end - first;
                    on_progress(completed as f32 / cells as f32);
                }
                first = end;
            }
        }
    }
    poll(should_continue)?;
    work.phases = profile.finish();
    Ok(Output {
        codes,
        frames,
        bars,
        work,
    })
}

#[test]
fn real_music_cqt_sync_matches_unchanged_old_code_bit_for_bit() {
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicU32};

    let path = Path::new(r"C:\Users\PC\Documents\ASYN\Audio Archive\Vocaloid\01_02_退廃的人生讃歌.flac");
    let cancel = AtomicBool::new(false);
    let prog = AtomicU32::new(0);
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut sr = 48_000u32;
    crate::measure::each_block(path, &cancel, &prog, |rate, ch, samples| {
        sr = rate;
        let start_frame = (rate as usize) * 30;
        let keep_frames = (rate as usize) * 3; // 3.0s real music excerpt
        for frame in samples.chunks(ch.max(1)) {
            if left.len() < start_frame + keep_frames {
                left.push(frame[0]);
                right.push(if ch > 1 { frame[1] } else { frame[0] });
            }
        }
    }).expect("decode real music");
    let start_f = (sr as usize) * 30;
    let l_slice = &left[start_f..];
    let r_slice = &right[start_f..];

    let bars = 128usize;
    let frequencies: Vec<f64> = (0..bars)
        .map(|i| 40.0 * (16_000.0_f64 / 40.0).powf((i as f64 + 0.5) / bars as f64))
        .collect();
    let cfg = Config {
        bins_per_octave: 24,
        window: Window::Blackman,
        frame_rate: 180.0,
        bass_limit_s: None,
    };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap();
    let input = Input::Joint { left: l_slice, right: r_slice };
    let (true_old_out, mod_ref_out, dyn_out, sync_out) = pool.install(|| {
        let to = true_de4fcb3_analyze_input_codes(
            input, sr, &frequencies, &cfg, Limits::default(), &|| true, &|_| {}, &code,
        ).unwrap();
        let mo = modified_serial_prep_ref_analyze_input_codes(
            input, sr, &frequencies, &cfg, Limits::default(), &|| true, &|_| {}, &code,
        ).unwrap();
        let d = analyze_input_codes(
            input, sr, &frequencies, &cfg, Limits::default(), &|| true, &|_| {}, &code,
        ).unwrap();
        let s = analyze_input_codes_sync(
            input, sr, &frequencies, &cfg, Limits::default(), &|| true, &|_| {}, &code,
        ).unwrap();
        (to, mo, d, s)
    });
    assert_eq!(sync_out.frames, true_old_out.frames);
    assert_eq!(sync_out.bars, true_old_out.bars);
    assert_eq!(sync_out.codes[0], true_old_out.codes[0], "Mix codes differ vs true de4fcb3 on real music!");
    assert_eq!(sync_out.codes[1], true_old_out.codes[1], "Left codes differ vs true de4fcb3 on real music!");
    assert_eq!(sync_out.codes[2], true_old_out.codes[2], "Right codes differ vs true de4fcb3 on real music!");
    assert_eq!(sync_out.codes, mod_ref_out.codes);
    assert_eq!(sync_out.codes, dyn_out.codes);
    assert_eq!(sync_out.work, dyn_out.work);
    assert_work_counts_eq(&mod_ref_out.work, &true_old_out.work);
    assert_work_counts_eq(&sync_out.work, &true_old_out.work);
}

#[test]
#[ignore = "same-binary A/B benchmark on real music (1024 bars @ 180 fps)"]
fn bench_cqt_ab_real_music() {
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicU32};
    use std::time::Instant;

    let path = Path::new(r"C:\Users\PC\Documents\ASYN\Audio Archive\Vocaloid\01_02_退廃的人生讃歌.flac");
    let cancel = AtomicBool::new(false);
    let prog = AtomicU32::new(0);
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut sr = 48_000u32;
    crate::measure::each_block(path, &cancel, &prog, |rate, ch, samples| {
        sr = rate;
        let target_len = (rate as usize) * 50;
        for frame in samples.chunks(ch.max(1)) {
            if left.len() < target_len {
                left.push(frame[0]);
                right.push(if ch > 1 { frame[1] } else { frame[0] });
            }
        }
    }).expect("decode real music");
    // Take 20.0s active vocal/instrumental section (30.0s .. 50.0s)
    let s0 = (sr as usize) * 30;
    let s1 = (sr as usize) * 50;
    let l20 = &left[s0..s1];
    let r20 = &right[s0..s1];
    let secs = l20.len() as f64 / sr as f64;

    let bars = 1024usize;
    let frequencies: Vec<f64> = (0..bars)
        .map(|i| 20.0 * (24_000.0_f64 / 20.0).powf((i as f64 + 0.5) / bars as f64))
        .collect();
    let cfg = Config::default(); // 48 bpo, Blackman, 180 fps
    let pool = rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap();
    let input = Input::Joint { left: l20, right: r20 };

    // Record Round 1 modified reference route (serial kernel prep + new Plan/Workspace) separately:
    let (d_mod_ref, mod_work) = pool.install(|| {
        let t0 = Instant::now();
        let mo = modified_serial_prep_ref_analyze_input_codes(
            input, sr, &frequencies, &cfg, Limits::default(), &|| true, &|_| {}, &code,
        ).unwrap();
        (t0.elapsed().as_secs_f64(), mo.work)
    });
    println!(
        "  [Round 1 modified reference route (serial kernel prep + new Plan/Workspace)]: {d_mod_ref:.3}s (work={mod_work:?})"
    );

    let mut old_s = Vec::new();
    let mut new_s = Vec::new();
    let mut verified_work = None;
    for pair in 0..3 {
        let (to, tn, order_str, o_work, n_work) = pool.install(|| {
            if pair % 2 == 0 {
                let t0 = Instant::now();
                let o = true_de4fcb3_analyze_input_codes(
                    input, sr, &frequencies, &cfg, Limits::default(), &|| true, &|_| {}, &code,
                ).unwrap();
                let d_old = t0.elapsed().as_secs_f64();

                let t1 = Instant::now();
                let n = analyze_input_codes_sync(
                    input, sr, &frequencies, &cfg, Limits::default(), &|| true, &|_| {}, &code,
                ).unwrap();
                let d_new = t1.elapsed().as_secs_f64();

                assert_eq!(n.codes, o.codes, "pair {}: CQT codes mismatch vs true de4fcb3!", pair + 1);
                assert_work_counts_eq(&n.work, &o.work);
                (d_old, d_new, "true_de4fcb3 -> new_sync", o.work, n.work)
            } else {
                let t1 = Instant::now();
                let n = analyze_input_codes_sync(
                    input, sr, &frequencies, &cfg, Limits::default(), &|| true, &|_| {}, &code,
                ).unwrap();
                let d_new = t1.elapsed().as_secs_f64();

                let t0 = Instant::now();
                let o = true_de4fcb3_analyze_input_codes(
                    input, sr, &frequencies, &cfg, Limits::default(), &|| true, &|_| {}, &code,
                ).unwrap();
                let d_old = t0.elapsed().as_secs_f64();

                assert_eq!(n.codes, o.codes, "pair {}: CQT codes mismatch vs true de4fcb3!", pair + 1);
                assert_work_counts_eq(&n.work, &o.work);
                (d_old, d_new, "new_sync -> true_de4fcb3", o.work, n.work)
            }
        });
        verified_work = Some((o_work, n_work));
        println!(
            "  pair {} ({order_str}): true_de4fcb3={to:.3}s, new_sync={tn:.3}s [3 planes x 3600x1024 BIT-EXACT, work counts matched]",
            pair + 1
        );
        old_s.push(to);
        new_s.push(tn);
    }
    let (ow, nw) = verified_work.unwrap();
    println!("  Verified WorkStats true_de4fcb3: {:?}", ow);
    println!("  Verified WorkStats new_sync:     {:?}", nw);
    old_s.sort_by(f64::total_cmp);
    new_s.sort_by(f64::total_cmp);
    let med_old = old_s[old_s.len() / 2];
    let med_new = new_s[new_s.len() / 2];
    let track216_old = med_old * (216.0 / secs);
    let track216_new = med_new * (216.0 / secs);
    println!(
        "=== MEASURED 20.0s excerpt (48kHz stereo, 1024 bars @ 180fps, 6 threads): true_de4fcb3={med_old:.3}s -> new_sync={med_new:.3}s ({:.2}x, -{:.1}%) ===",
        med_old / med_new, (1.0 - med_new / med_old) * 100.0
    );
    println!(
        "  ESTIMATE (linear extrapolation to 216s track from 20.0s excerpt; not full-track measured): true_de4fcb3~={track216_old:.2}s -> new_sync~={track216_new:.2}s (estimated saving ~{:.2}s)",
        track216_old - track216_new
    );
}

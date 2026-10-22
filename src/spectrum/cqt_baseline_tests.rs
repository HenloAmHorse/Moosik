//! Independent frozen-prereview gate, separate from the candidate's helpers.

use crate::spectrum::cqt;
use std::cell::RefCell;

type ComplexBits = ([u64; 2], Option<[u64; 2]>);
type ObservedBits = RefCell<Vec<Option<ComplexBits>>>;

// The immutable baseline contains its own public UI helpers and result fields.
// They are deliberately retained verbatim rather than deleting unused pieces.
#[allow(dead_code)]
#[path = "../../research/v155/frozen_cqt_f9a.rs"]
mod frozen;

fn old_config(cfg: cqt::Config) -> frozen::Config {
    frozen::Config {
        bins_per_octave: cfg.bins_per_octave,
        frame_rate: cfg.frame_rate,
        window: match cfg.window {
            cqt::Window::Hann => frozen::Window::Hann,
            cqt::Window::Hamming => frozen::Window::Hamming,
            cqt::Window::Blackman => frozen::Window::Blackman,
            cqt::Window::BlackmanHarris => frozen::Window::BlackmanHarris,
        },
    }
}

fn samples(sr: u32, count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.12;
            let t = i as f64 / sr as f64;
            if i == 0 {
                -0.0
            } else if i == count / 2 {
                0.8
            } else {
                (0.55 * (std::f64::consts::TAU * (77.0 * t + 1300.0 * t * t)).sin() + noise) as f32
            }
        })
        .collect()
}

#[test]
fn cqt_candidate_matches_the_frozen_prereview_code_planes() {
    let frequencies = [90.0, 750.0, 4200.0, 12000.0];
    for sr in [44100, 48000, 96000] {
        let left = samples(sr, 1537, 0x0c00_1155);
        for (index, window) in [
            cqt::Window::Hann,
            cqt::Window::Hamming,
            cqt::Window::Blackman,
            cqt::Window::BlackmanHarris,
        ]
        .into_iter()
        .enumerate()
        {
            let cfg = cqt::Config {
                bins_per_octave: 24,
                window,
                frame_rate: [180.0, 179.5, 137.5, 360.0][index],
                bass_limit_s: None,
            };
            let old_cfg = old_config(cfg);
            let right = if index % 2 == 1 {
                left.iter().map(|x| -*x).collect()
            } else {
                samples(sr, left.len(), 0x900d_cafe)
            };
            for joint in [false, true] {
                let old_input = if joint {
                    frozen::Input::Joint {
                        left: &left,
                        right: &right,
                    }
                } else {
                    frozen::Input::Mono(&left)
                };
                let input = if joint {
                    cqt::Input::Joint {
                        left: &left,
                        right: &right,
                    }
                } else {
                    cqt::Input::Mono(&left)
                };
                let cells =
                    cqt::frame_count(left.len(), sr, cfg.frame_rate).unwrap() * frequencies.len();
                let old_bits = RefCell::new(vec![None; cells]);
                let new_bits = RefCell::new(vec![None; cells]);
                let save =
                    |slots: &ObservedBits,
                     frame: usize,
                     bar: usize,
                     l: rustfft::num_complex::Complex<f64>,
                     r: Option<rustfft::num_complex::Complex<f64>>| {
                        let bits = (
                            [l.re.to_bits(), l.im.to_bits()],
                            r.map(|v| [v.re.to_bits(), v.im.to_bits()]),
                        );
                        assert!(
                            slots.borrow_mut()[frame * frequencies.len() + bar]
                                .replace(bits)
                                .is_none()
                        );
                    };
                let old_observer = |frame, bar, l, r| save(&old_bits, frame, bar, l, r);
                let new_observer = |frame, bar, l, r| save(&new_bits, frame, bar, l, r);
                let old = frozen::analyze_observed(
                    old_input,
                    sr,
                    &frequencies,
                    &old_cfg,
                    frozen::Limits::default(),
                    &crate::spectrum::aslt_code,
                    &old_observer,
                )
                .unwrap();
                let new = cqt::analyze_inner(
                    input,
                    sr,
                    &frequencies,
                    &cfg,
                    cqt::Limits::default(),
                    &|| true,
                    &|_| {},
                    &crate::spectrum::aslt_code,
                    Some(&new_observer),
                    false,
                )
                .unwrap();
                assert_eq!(new.codes, old.codes, "sr={sr} config={cfg:?} joint={joint}");
                assert!(old_bits.borrow().iter().all(Option::is_some));
                assert_eq!(
                    new_bits.into_inner(),
                    old_bits.into_inner(),
                    "frozen complex bits changed sr={sr} config={cfg:?} joint={joint}"
                );
                assert_eq!((new.frames, new.bars), (old.frames, old.bars));
                assert_eq!(new.work.column_convolutions, old.work.column_convolutions);
            }
        }
    }
}

#[test]
fn frozen_cqt_source_is_the_committed_prereview_not_candidate_helpers() {
    let text = include_str!("../../research/v155/frozen_cqt_f9a.rs");
    assert!(text.contains("f9a69999d8d97d032e2f3c07ecc204a65ed50369"));
    assert!(!text.contains("process_pruned"));
    assert!(!text.contains("super::cqt"));
    let normalized = text.replace("\r\n", "\n");
    let original = normalized
        .split_once("\n// BEGIN TEST ACCESSORS\n")
        .unwrap()
        .0;
    let digest = ring::digest::digest(&ring::digest::SHA256, original.as_bytes());
    let hex: String = digest.as_ref().iter().map(|v| format!("{v:02x}")).collect();
    assert_eq!(
        hex,
        "877d657204f933f17007559e5fef7596c22b25fe3f815d159e35ff7333d504ab"
    );
}

#[test]
fn explicit_worker_pools_and_tight_batches_preserve_frozen_complex_bits() {
    let sr = 48000;
    let cfg = cqt::Config::default();
    let frequencies: Vec<_> = (0..33).map(|i| 500.0 + i as f64).collect();
    let left = samples(sr, 20001, 0x155155);
    let right = samples(sr, left.len(), 0x778899);
    let cells = cqt::frame_count(left.len(), sr, cfg.frame_rate).unwrap() * frequencies.len();
    let old_bits = RefCell::new(vec![None; cells]);
    let old = frozen::analyze_observed(
        frozen::Input::Joint {
            left: &left,
            right: &right,
        },
        sr,
        &frequencies,
        &old_config(cfg),
        frozen::Limits::default(),
        &crate::spectrum::aslt_code,
        &|frame, bar, l, r| {
            old_bits.borrow_mut()[frame * frequencies.len() + bar] = Some((
                [l.re.to_bits(), l.im.to_bits()],
                r.map(|v| [v.re.to_bits(), v.im.to_bits()]),
            ));
        },
    )
    .unwrap();
    for workers in [1, 2, 6] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        for budget in [2 << 20, 8 << 20, 64 << 20] {
            let (new, bits) = pool.install(|| {
                let bits = RefCell::new(vec![None; cells]);
                let output = cqt::analyze_inner(
                    cqt::Input::Joint {
                        left: &left,
                        right: &right,
                    },
                    sr,
                    &frequencies,
                    &cfg,
                    cqt::Limits {
                        workspace_bytes: budget,
                        ..Default::default()
                    },
                    &|| true,
                    &|_| {},
                    &crate::spectrum::aslt_code,
                    Some(&|frame, bar, l, r| {
                        assert!(
                            bits.borrow_mut()[frame * frequencies.len() + bar]
                                .replace((
                                    [l.re.to_bits(), l.im.to_bits()],
                                    r.map(|v| [v.re.to_bits(), v.im.to_bits()]),
                                ))
                                .is_none()
                        );
                    }),
                    false,
                )
                .unwrap();
                (output, bits.into_inner())
            });
            assert_eq!(new.codes, old.codes, "workers={workers} budget={budget}");
            assert_eq!(
                bits,
                *old_bits.borrow(),
                "complex workers={workers} budget={budget}"
            );
            assert!(new.work.peak_workspace_bytes <= budget);
            assert!(new.work.max_inverse_workers <= workers);
            if workers > 1 && budget == 64 << 20 {
                assert!(
                    new.work.parallel_groups > 0,
                    "parallel route was not exercised"
                );
                assert_eq!(new.work.max_inverse_workers, workers);
            }
            if workers == 1 || budget == 2 << 20 {
                assert_eq!(new.work.parallel_groups, 0);
            }
        }
    }
}

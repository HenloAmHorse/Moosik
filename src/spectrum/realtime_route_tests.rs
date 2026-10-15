//! The realtime bar route against a frozen copy of what it was.
//!
//! `reference_bins_to_bars` is `SpectrumAnalyzer::bins_to_bars` exactly as it
//! stood at `b9b37a2`, before any realtime change in batch 1 — copied, not
//! refactored, so the production route is never checked against a
//! rearrangement of itself. Every comparison here is of `f32` bits.

use super::*;

/// Frozen at `b9b37a2`. Do not edit: it is the reference.
#[allow(clippy::all)]
pub(super) fn reference_bins_to_bars(a: &SpectrumAnalyzer, fft_out: &[Complex<f32>], sr: u32, n_bars: usize, padded_size: usize) -> Vec<f32> {
    let fft_size = a.fft_size;
    let half = padded_size / 2;
    let log_min = a.min_freq.log10();
    let log_max = (sr as f32 / 2.0).min(a.max_freq).log10();
    let norms: Vec<f32> = fft_out[..half].iter().map(|c| c.norm()).collect();
    let scale = fft_size as f32; // normalize by window length, not padded length

    // Pre-compute Q factor for CQT mode: Q = 1 / (2^(1/B) - 1)
    // where B = bars per octave = n_bars / log2(fmax/fmin)
    let n_octaves = (log_max - log_min) / 2_f32.log10();
    let bins_per_oct = n_bars as f32 / n_octaves.max(0.1);
    let cqt_q = 1.0 / (2.0_f32.powf(1.0 / bins_per_oct) - 1.0);

    (0..n_bars)
        .map(|bar| {
            // Bar edges follow whichever scale the display is on, or the
            // FFT bins would be gathered into bars that sit somewhere else.
            let fscale = a.aslt_cfg.scale;
            let (lo_hz, hi_hz) = (a.min_freq, a.max_freq);
            let mag = if a.bar_mapping == BarMappingMode::Cqt {
                // CQT: centre-frequency Hann kernel with constant-Q bandwidth
                let tc  = (bar as f32 + 0.5) / n_bars as f32;
                let f_c = fscale.freq_at(tc, lo_hz, hi_hz);
                let bc  = (f_c * padded_size as f32 / sr as f32).clamp(1.0, half as f32 - 1.0);
                cqt_kernel(&norms, bc, cqt_q, half)
            } else {
                let t0 = bar as f32 / n_bars as f32;
                let t1 = (bar + 1) as f32 / n_bars as f32;
                let freq_lo = fscale.freq_at(t0, lo_hz, hi_hz);
                let freq_hi = fscale.freq_at(t1, lo_hz, hi_hz);
                // Both ends are bounded by the spectrum that exists.
                //
                // `max_freq` defaults to 24 kHz, which is above Nyquist for
                // a 44.1 kHz track, so the top bars ask for bins the FFT
                // never produced. Only `fbin_hi` used to be clamped; a bar
                // whose *low* edge was already past the end then produced a
                // negative width, took the sub-bin branch, and indexed the
                // midpoint of a range that started off the end of the array.
                // Those bars read the topmost bin instead, which is what the
                // constant-Q branch has always done with them.
                let top = half as f32 - 1.0;
                let fbin_lo = (freq_lo * padded_size as f32 / sr as f32).clamp(1.0, top);
                let fbin_hi = (freq_hi * padded_size as f32 / sr as f32)
                    .clamp(fbin_lo, top);

                // Sub-bin: interpolated; Multi-bin: weighted overlap average.
                if fbin_hi - fbin_lo <= 1.0 {
                    let center = (fbin_lo + fbin_hi) * 0.5;
                    interp_sub_bin(&norms, center, &a.interp_mode)
                } else {
                    let b_start = fbin_lo.floor() as usize;
                    let b_end   = (fbin_hi.ceil() as usize).min(half - 1);
                    let bc      = (fbin_lo + fbin_hi) * 0.5;
                    let sigma   = ((fbin_hi - fbin_lo) * 0.5).max(0.5);
                    let mut wsum = 0.0_f32;
                    let mut weight = 0.0_f32;
                    for (b_idx, &norm_b) in norms[b_start..=b_end].iter().enumerate() {
                        let b = b_start + b_idx;
                        let w = match &a.bar_mapping {
                            BarMappingMode::FlatOverlap => {
                                (fbin_hi.min(b as f32 + 1.0) - fbin_lo.max(b as f32)).max(0.0)
                            }
                            BarMappingMode::Gaussian
                            | BarMappingMode::Cqt
                            | BarMappingMode::Superlet => {
                                let center_b = b as f32 + 0.5;
                                (-(center_b - bc).powi(2) / (2.0 * sigma * sigma)).exp()
                            }
                        };
                        wsum += norm_b * w; weight += w;
                    }
                    if weight > 0.0 { wsum / weight } else { 0.0 }
                }
            };

            let raw_db = 20.0 * (mag / scale).log10().max(-80.0);
            let corrected_db = if let Some(&w) = a.eq_weights.get(bar) { raw_db + w } else { raw_db };
            ((corrected_db + 80.0) / 80.0).clamp(0.0, 1.0)
        })
        .collect()
}

fn spectrum(padded: usize, seed: u32) -> Vec<Complex<f32>> {
    let mut x = seed.max(1);
    let mut r = || { x ^= x << 13; x ^= x >> 17; x ^= x << 5; x as f32 / u32::MAX as f32 };
    (0..padded).map(|i| {
        let tilt = 40.0 / (1.0 + i as f32).sqrt();
        Complex { re: (r() - 0.5) * tilt * padded as f32, im: (r() - 0.5) * tilt * padded as f32 }
    }).collect()
}

/// The configurations the display can reach that change the bar arithmetic.
/// A named way to set an analyser up.
pub(super) type Config = (String, Box<dyn Fn(&mut SpectrumAnalyzer)>);

pub(super) fn configs() -> Vec<Config> {
    let mut v: Vec<Config> = Vec::new();
    let scales = [
        ("log", freq_scale::FreqScale::Log),
        ("erb", freq_scale::FreqScale::Erb),
        ("blend", freq_scale::FreqScale::Blend(-0.35)),
        ("lens", freq_scale::FreqScale::from_parts(0.2, 900.0, 1.7, 2.5)),
    ];
    let mappings = [BarMappingMode::FlatOverlap, BarMappingMode::Gaussian, BarMappingMode::Cqt];
    let interps = [InterpolationMode::None, InterpolationMode::Linear, InterpolationMode::CatmullRom,
                   InterpolationMode::Pchip, InterpolationMode::Akima, InterpolationMode::Lanczos];
    for (sn, sc) in scales {
        for m in &mappings {
            for (ii, i) in interps.iter().enumerate() {
                if ii > 0 && sn != "log" { continue; }
                for (bars, sr, fft, lo, hi, eq) in [
                    (64usize, 44_100u32, 2048usize, 20.0f32, 24_000.0f32, false),
                    (1024, 48_000, 8192, 20.0, 20_000.0, true),
                    (333, 96_000, 16_384, 35.5, 30_000.0, false),
                ] {
                    let (sc, m, i) = (sc, m.clone(), i.clone());
                    v.push((format!("{sn} {m:?} {i:?} {bars} bars {sr} Hz fft {fft} eq {eq}"), Box::new(move |a: &mut SpectrumAnalyzer| {
                        a.aslt_cfg.scale = sc;
                        a.bar_mapping = m.clone();
                        a.interp_mode = i.clone();
                        a.bar_count = bars;
                        a.sample_rate = sr;
                        a.live_rate = sr;
                        a.fft_size = fft;
                        a.min_freq = lo;
                        a.max_freq = hi;
                        a.eq_weights = if eq { compute_eq_weights(bars, lo, hi, sc) } else { Vec::new() };
                    })));
                }
            }
        }
    }
    v
}

fn bits(v: &[f32]) -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() }

/// The split `bins_to_bars` — norms, then `bars_from_norms`, then `bar_db` —
/// is the frozen function, bit for bit, everywhere it can be driven.
#[test]
fn the_split_bar_mapping_is_the_frozen_one() {
    let mut n = 0;
    for (what, set) in configs() {
        let mut a = SpectrumAnalyzer::new(new_sample_buf());
        set(&mut a);
        let padded = a.fft_size * 2;
        for seed in [1u32, 7, 99] {
            let s = spectrum(padded, seed);
            let got = a.bins_to_bars(&s, a.sample_rate, a.bar_count, padded);
            let want = reference_bins_to_bars(&a, &s, a.sample_rate, a.bar_count, padded);
            assert_eq!(bits(&got), bits(&want), "{what} seed {seed}");
            n += 1;
        }
    }
    assert!(n > 100, "setup: only {n} cases");
}

/// Audio for `ticks` realtime frames: a chirp, two tones and noise, so every
/// bar moves and the smoother never settles into a fixed point.
fn audio(sr: u32, len: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.max(1);
    (0..len).map(|i| {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        let t = i as f32 / sr as f32;
        let noise = (x as f32 / u32::MAX as f32) * 2.0 - 1.0;
        0.3 * (std::f32::consts::TAU * (80.0 + 900.0 * t) * t).sin()
            + 0.2 * (std::f32::consts::TAU * 440.0 * t).sin()
            + 0.1 * (std::f32::consts::TAU * 7_000.0 * t).sin()
            + 0.05 * noise
    }).collect()
}

/// Everything a realtime Mix tick leaves behind for the display.
#[derive(PartialEq, Debug)]
pub(super) struct TickState {
    magnitudes: Vec<u32>,
    smoothed: Vec<u32>,
    peak_input: Vec<u32>,
    norms: Vec<u32>,
    waterfall: Vec<Vec<u32>>,
    alpha: Option<u32>,
}

pub(super) fn state(a: &SpectrumAnalyzer) -> TickState {
    TickState {
        magnitudes: bits(&a.magnitudes),
        smoothed: bits(&a.smoothed),
        peak_input: bits(&a.peak_input),
        norms: bits(&a.last_fft_norms),
        waterfall: a.waterfall.iter().map(|r| bits(r)).collect(),
        alpha: a.frame_alpha.map(f32::to_bits),
    }
}

/// An analyser set up by `set`, ready for realtime ticks.
pub(super) fn rt_analyzer(set: &dyn Fn(&mut SpectrumAnalyzer), reference: bool) -> SpectrumAnalyzer {
    let mut a = SpectrumAnalyzer::new(new_sample_buf());
    set(&mut a);
    a.rebuild_fft();
    a.magnitudes = vec![0.0; a.bar_count];
    a.smoothed = vec![0.0; a.bar_count];
    a.peak_input = vec![0.0; a.bar_count];
    a.waterfall_enabled = true;
    a.smoothing = 0.6;
    a.rt_reference = reference;
    a
}

/// Feed both analysers the same audio, tick by tick, and require identical
/// downstream state after every tick.
pub(super) fn drive_pair(what: &str, a: &mut SpectrumAnalyzer, b: &mut SpectrumAnalyzer, ticks: usize) {
    let hop = a.fft_size / 3;
    let pcm = audio(a.sample_rate, a.fft_size + hop * ticks, 11);
    for k in 0..ticks {
        let window = &pcm[..a.fft_size + hop * k];
        for x in [&mut *a, &mut *b] {
            let mut g = x.sample_buf.lock().unwrap();
            g.clear();
            g.extend_from_slice(window);
        }
        a.process_realtime(1.0 / 60.0);
        b.process_realtime(1.0 / 60.0);
        assert_eq!(state(a), state(b), "{what}: tick {k}");
    }
    // Not a comparison of two empty displays.
    assert!(a.magnitudes.iter().any(|&m| m > 0.0), "{what}: setup: silent bars");
    assert!(a.waterfall.len() >= ticks, "{what}: setup: no waterfall history");
}

/// The comparison can fail: one equal-loudness weight nudged by 0.0001 dB on
/// one side must show up in the state. (One ulp of a small weight is not a
/// usable control: the dB sum it joins rounds it away.)
#[test]
fn the_downstream_comparison_detects_a_tiny_difference() {
    let (what, set) = configs().into_iter().find(|(w, _)| w.contains("eq true")).expect("setup");
    let mut a = rt_analyzer(&*set, false);
    let mut b = rt_analyzer(&*set, true);
    b.eq_weights[b.bar_count / 2] += 1e-4;
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drive_pair(&what, &mut a, &mut b, 3)));
    assert!(r.is_err(), "a 0.0001 dB difference went unnoticed");
}

/// The production realtime Mix tick against the same tick through the frozen
/// route: magnitudes, smoother, peak input, FFT norms, frame alpha and the
/// waterfall history are bit-identical after every tick.
#[test]
fn the_realtime_tick_matches_the_frozen_route_downstream() {
    for (what, set) in configs() {
        let mut prod = rt_analyzer(&*set, false);
        let mut frozen = rt_analyzer(&*set, true);
        drive_pair(&what, &mut prod, &mut frozen, 6);
    }
}

/// The Mix/constant-Q realtime tick uses the plan, keeps it while the
/// geometry holds, and rebuilds it — once — whenever any field the plan reads
/// changes, independently of the others, and again on the way back. After
/// every step the tick equals the frozen route downstream.
#[test]
fn the_constant_q_plan_is_used_and_follows_every_field_it_reads() {
    use freq_scale::FreqScale as F;
    let base = |a: &mut SpectrumAnalyzer| {
        a.bar_mapping = BarMappingMode::Cqt;
        a.aslt_cfg.scale = F::from_parts(0.2, 900.0, 1.7, 2.5);
        a.bar_count = 256;
        a.sample_rate = 48_000;
        a.live_rate = 48_000;
        a.fft_size = 4096;
        a.min_freq = 25.0;
        a.max_freq = 20_000.0;
    };
    type Edit = fn(&mut SpectrumAnalyzer);
    let edits: [(&str, Edit); 12] = [
        ("bars", |a| a.bar_count = 255),
        ("fft size", |a| { a.fft_size = 8192; a.rebuild_fft(); }),
        ("live rate", |a| a.live_rate = 44_100),
        ("min freq", |a| a.min_freq = 25.5),
        ("max freq", |a| a.max_freq = 19_000.0),
        ("max freq past Nyquist", |a| a.max_freq = 30_000.0),
        ("scale kind", |a| a.aslt_cfg.scale = F::Erb),
        ("blend tilt", |a| a.aslt_cfg.scale = F::Blend(0.3)),
        ("lens tilt", |a| a.aslt_cfg.scale = F::from_parts(0.25, 900.0, 1.7, 2.5)),
        ("lens hz", |a| a.aslt_cfg.scale = F::from_parts(0.2, 901.0, 1.7, 2.5)),
        ("lens oct", |a| a.aslt_cfg.scale = F::from_parts(0.2, 900.0, 1.8, 2.5)),
        ("lens gain", |a| a.aslt_cfg.scale = F::from_parts(0.2, 900.0, 1.7, 2.6)),
    ];
    for (what, edit) in edits {
        let mut prod = rt_analyzer(&base, false);
        let mut frozen = rt_analyzer(&base, true);
        drive_pair(&format!("{what}: base"), &mut prod, &mut frozen, 3);
        assert_eq!(prod.cqt_plan_builds, 1, "{what}: the plan was not built once and kept");
        for x in [&mut prod, &mut frozen] { edit(x); }
        drive_pair(&format!("{what}: changed"), &mut prod, &mut frozen, 3);
        assert_eq!(prod.cqt_plan_builds, 2, "{what}: the change did not rebuild the plan exactly once");
        for x in [&mut prod, &mut frozen] {
            base(x);
            x.rebuild_fft();
        }
        drive_pair(&format!("{what}: back"), &mut prod, &mut frozen, 3);
        assert_eq!(prod.cqt_plan_builds, 3, "{what}: returning did not rebuild the plan");
    }
}

/// What the plan does not read does not rebuild it: interpolation, the
/// equal-loudness weights and the smoothing change the result or not, but
/// never the gather. And no other mapping builds a plan at all.
#[test]
fn the_constant_q_plan_ignores_what_it_does_not_read_and_other_mappings_skip_it() {
    let base = |a: &mut SpectrumAnalyzer| {
        a.bar_mapping = BarMappingMode::Cqt;
        a.bar_count = 128;
        a.sample_rate = 44_100;
        a.live_rate = 44_100;
        a.fft_size = 2048;
    };
    let mut prod = rt_analyzer(&base, false);
    let mut frozen = rt_analyzer(&base, true);
    drive_pair("cqt base", &mut prod, &mut frozen, 2);
    for x in [&mut prod, &mut frozen] {
        x.interp_mode = InterpolationMode::Akima;
        x.eq_weights = compute_eq_weights(x.bar_count, x.min_freq, x.max_freq, x.aslt_cfg.scale);
        x.smoothing = 0.2;
    }
    drive_pair("cqt unrelated changes", &mut prod, &mut frozen, 2);
    assert_eq!(prod.cqt_plan_builds, 1, "an unrelated setting rebuilt the plan");
    for m in [BarMappingMode::FlatOverlap, BarMappingMode::Gaussian] {
        let mut a = rt_analyzer(&|a: &mut SpectrumAnalyzer| { base(a); a.bar_mapping = m.clone(); }, false);
        let mut b = rt_analyzer(&|a: &mut SpectrumAnalyzer| { base(a); a.bar_mapping = m.clone(); }, true);
        drive_pair(&format!("{m:?}"), &mut a, &mut b, 2);
        assert_eq!(a.cqt_plan_builds, 0, "{m:?} built a constant-Q plan");
    }
}

/// The blend tilt alone, Blend to Blend: the kind is the same, only the
/// parameter moves, and the plan must still be rebuilt.
#[test]
fn the_constant_q_plan_follows_a_blend_tilt_alone() {
    let at = |tilt: f32| move |a: &mut SpectrumAnalyzer| {
        a.bar_mapping = BarMappingMode::Cqt;
        a.aslt_cfg.scale = freq_scale::FreqScale::Blend(tilt);
        a.bar_count = 200;
        a.sample_rate = 44_100;
        a.live_rate = 44_100;
        a.fft_size = 4096;
    };
    let mut prod = rt_analyzer(&at(0.3), false);
    let mut frozen = rt_analyzer(&at(0.3), true);
    drive_pair("blend 0.3", &mut prod, &mut frozen, 2);
    for x in [&mut prod, &mut frozen] { at(0.35)(x); }
    drive_pair("blend 0.35", &mut prod, &mut frozen, 2);
    assert_eq!(prod.cqt_plan_builds, 2, "a blend tilt change did not rebuild the plan");
}

/// The Left/Right tick under every configuration: after each Mix tick, the
/// channel tick on the same stereo audio, production against the frozen
/// route. Both channels' bars and smoothers are bit-identical, and under the
/// constant-Q mapping the channels share the Mix's plan — no second build.
#[test]
fn the_channel_tick_matches_the_frozen_route_and_shares_the_plan() {
    for (what, set) in configs() {
        let mut prod = rt_analyzer(&*set, false);
        let mut frozen = rt_analyzer(&*set, true);
        let hop = prod.fft_size / 3;
        let len = prod.fft_size + hop * 4;
        let l = audio(prod.sample_rate, len, 21);
        let r = audio(prod.sample_rate, len, 22);
        for k in 0..4 {
            let end = prod.fft_size + hop * k;
            for x in [&mut prod, &mut frozen] {
                let mut g = x.sample_buf.lock().unwrap();
                g.clear();
                g.extend(l[..end].iter().zip(&r[..end]).map(|(a, b)| (a + b) * 0.5));
            }
            for x in [&mut prod, &mut frozen] {
                x.process_realtime(1.0 / 60.0);
                x.process_channels(&l[..end], &r[..end], 1.0 / 60.0);
            }
            let ch = |a: &SpectrumAnalyzer| (bits(&a.bars_left), bits(&a.bars_right),
                                            bits(&a.smoothed_left), bits(&a.smoothed_right));
            assert_eq!(ch(&prod), ch(&frozen), "{what}: tick {k}");
            assert_eq!(state(&prod), state(&frozen), "{what}: mix, tick {k}");
        }
        assert!(prod.bars_left.iter().any(|&b| b > 0.0), "{what}: setup: silent channels");
        let want = if what.contains("Cqt") { 1 } else { 0 };
        assert_eq!(prod.cqt_plan_builds, want, "{what}: the channels built their own plan");
    }
}

/// The channel tick checks the plan itself: after a geometry change, a
/// channel tick that runs before any Mix tick must not gather from the old
/// geometry's plan.
#[test]
fn the_channel_tick_does_not_trust_a_plan_from_another_geometry() {
    let at = |bars: usize| move |a: &mut SpectrumAnalyzer| {
        a.bar_mapping = BarMappingMode::Cqt;
        a.bar_count = bars;
        a.sample_rate = 44_100;
        a.live_rate = 44_100;
        a.fft_size = 2048;
    };
    let mut prod = rt_analyzer(&at(96), false);
    let mut frozen = rt_analyzer(&at(96), true);
    let l = audio(44_100, 6_000, 31);
    let r = audio(44_100, 6_000, 32);
    for x in [&mut prod, &mut frozen] {
        x.process_channels(&l, &r, 1.0 / 60.0);
        at(97)(x);
        x.process_channels(&l, &r, 1.0 / 60.0);
    }
    assert_eq!(prod.bars_left.len(), 97);
    assert_eq!(bits(&prod.bars_left), bits(&frozen.bars_left));
    assert_eq!(bits(&prod.bars_right), bits(&frozen.bars_right));
    assert_eq!(prod.cqt_plan_builds, 2);
}
